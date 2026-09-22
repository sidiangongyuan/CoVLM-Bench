import torch
import torch.nn as nn
from mmdet.models.builder import HEADS, build_loss
from einops import rearrange
from projects.mmdet3d_plugin.models.utils.functional import bivariate_gaussian_activation
from .planning_head_plugin import CollisionNonlinearOptimizer
import numpy as np
import copy
import heapq

@HEADS.register_module()
class PlanningHeadSingleMode(nn.Module):
    def __init__(self,
                 bev_h=200,
                 bev_w=200,
                 embed_dims=256,
                 planning_steps=6,
                 loss_planning=None,
                 loss_collision=None,
                 planning_eval=False,
                 use_col_optim=False,
                 col_optim_args=dict(
                    occ_filter_range=5.0,
                    sigma=1.0, 
                    alpha_collision=5.0,
                 ),
                 with_adapter=False,
                 occ_n_future_only_occ=4,
                 num_commands=3,
                 predict_command=False,
                 loss_command=None,
                ):
        """
        Single Mode Planning Head for Autonomous Driving.

        Args:
            embed_dims (int): Embedding dimensions. Default: 256.
            planning_steps (int): Number of steps for motion planning. Default: 6.
            loss_planning (dict): Configuration for planning loss. Default: None.
            loss_collision (dict): Configuration for collision loss. Default: None.
            planning_eval (bool): Whether to use planning for evaluation. Default: False.
            use_col_optim (bool): Whether to use collision optimization. Default: False.
            col_optim_args (dict): Collision optimization arguments. Default: dict(occ_filter_range=5.0, sigma=1.0, alpha_collision=5.0).
        """
        super(PlanningHeadSingleMode, self).__init__()

        self.occ_n_future_only_occ = occ_n_future_only_occ
        # Nuscenes
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_commands = num_commands
        self.predict_command = predict_command
        self.navi_embed = nn.Embedding(num_commands, embed_dims)
        self.command_head = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, num_commands),
        )
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, planning_steps * 2),
        )
        self.loss_planning = build_loss(loss_planning)
        if loss_command is not None:
            self.loss_command = build_loss(loss_command)
        else:
            self.loss_command = nn.CrossEntropyLoss()
        self.planning_steps = planning_steps
        self.planning_eval = planning_eval
        
        #### planning head
        fuser_dim = 3
        attn_module_layer = nn.TransformerDecoderLayer(embed_dims, 8, dim_feedforward=embed_dims*2, dropout=0.1, batch_first=False)
        self.attn_module = nn.TransformerDecoder(attn_module_layer, 3)
        
        self.mlp_fuser = nn.Sequential(
                nn.Linear(embed_dims*fuser_dim, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
            )
        
        self.pos_embed = nn.Embedding(1, embed_dims)
        self.loss_collision = []
        for cfg in loss_collision:
            self.loss_collision.append(build_loss(cfg))
        self.loss_collision = nn.ModuleList(self.loss_collision)
        
        self.use_col_optim = use_col_optim
        self.occ_filter_range = col_optim_args['occ_filter_range']
        self.sigma = col_optim_args['sigma']
        self.alpha_collision = col_optim_args['alpha_collision']

        # TODO: reimplement it with down-scaled feature_map
        self.with_adapter = with_adapter
        if with_adapter:
            bev_adapter_block = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1),
            )
            N_Blocks = 3
            bev_adapter = [copy.deepcopy(bev_adapter_block) for _ in range(N_Blocks)]
            self.bev_adapter = nn.Sequential(*bev_adapter)
        self.num = 0

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        navi_key = prefix + 'navi_embed.weight'
        if navi_key in state_dict:
            saved = state_dict[navi_key]
            current = self.navi_embed.weight
            if saved.shape == torch.Size([3, current.shape[1]]) and current.shape[0] == 7:
                migrated = current.detach().clone()
                # Old UniV2X command ids: 0 RIGHT, 1 LEFT, 2 FORWARD.
                migrated[0] = saved[2]  # GO_STRAIGHT
                migrated[1] = saved[1]  # TURN_LEFT
                migrated[2] = saved[0]  # TURN_RIGHT
                migrated[3] = saved[2]  # LATERAL_SHIFT starts from forward prior.
                migrated[4] = saved[2]  # STOP starts from forward prior.
                migrated[5] = saved[2]  # SLOW_DOWN starts from forward prior.
                migrated[6] = saved[2]  # UNKNOWN starts from forward prior.
                state_dict[navi_key] = migrated
        reg_weight_key = prefix + 'reg_branch.2.weight'
        if reg_weight_key in state_dict:
            saved = state_dict[reg_weight_key]
            current = self.reg_branch[2].weight
            if saved.shape != current.shape and saved.shape[1:] == current.shape[1:]:
                if saved.shape[0] >= current.shape[0]:
                    state_dict[reg_weight_key] = saved[:current.shape[0]].clone()
        reg_bias_key = prefix + 'reg_branch.2.bias'
        if reg_bias_key in state_dict:
            saved = state_dict[reg_bias_key]
            current = self.reg_branch[2].bias
            if saved.shape != current.shape and saved.shape[0] >= current.shape[0]:
                state_dict[reg_bias_key] = saved[:current.shape[0]].clone()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def _command_to_tensor(self, command, device, batch_size=None):
        if command is None:
            return None
        if isinstance(command, (list, tuple)):
            parts = [self._command_to_tensor(each, device, None) for each in command]
            parts = [each for each in parts if each is not None]
            if not parts:
                return None
            command = torch.cat([each.reshape(-1) for each in parts], dim=0)
        elif torch.is_tensor(command):
            command = command.to(device=device)
        else:
            command = torch.tensor(command, device=device)
        command = command.long().reshape(-1)
        if batch_size is not None and command.numel() == 1 and batch_size > 1:
            command = command.expand(batch_size)
        if batch_size is not None:
            command = command[:batch_size]
        return command

    def _batch_to_tensor(self, value, device):
        if value is None:
            return None
        if torch.is_tensor(value):
            tensor = value
        elif isinstance(value, (list, tuple)):
            parts = [self._batch_to_tensor(each, device) for each in value]
            parts = [each for each in parts if each is not None]
            if not parts:
                return None
            tensor = torch.stack(parts, dim=0)
        else:
            tensor = torch.as_tensor(value)
        return tensor.to(device=device)

    def forward_train(self,
                      bev_embed, 
                      outs_motion={}, 
                      sdc_planning=None, 
                      sdc_planning_mask=None,
                      command=None,
                      gt_future_boxes=None,
                      ):
        """
        Perform forward planning training with the given inputs.
        Args:
            bev_embed (torch.Tensor): The input bird's eye view feature map.
            outs_motion (dict): A dictionary containing the motion outputs.
            outs_occflow (dict): A dictionary containing the occupancy flow outputs.
            sdc_planning (torch.Tensor, optional): The self-driving car's planned trajectory.
            sdc_planning_mask (torch.Tensor, optional): The mask for the self-driving car's planning.
            command (torch.Tensor, optional): The driving command issued to the self-driving car.
            gt_future_boxes (torch.Tensor, optional): The ground truth future bounding boxes.
            img_metas (list[dict], optional): A list of metadata information about the input images.

        Returns:
            ret_dict (dict): A dictionary containing the losses and planning outputs.
        """
        sdc_traj_query = outs_motion['sdc_traj_query']
        sdc_track_query = outs_motion['sdc_track_query']
        bev_pos = outs_motion['bev_pos']

        occ_mask = None
        
        outs_planning = self(bev_embed, occ_mask, bev_pos, sdc_traj_query, sdc_track_query, command, None)
        loss_inputs = [sdc_planning, sdc_planning_mask, outs_planning, gt_future_boxes, command]
        losses = self.loss(*loss_inputs)
        ret_dict = dict(losses=losses, outs_motion=outs_planning)
        return ret_dict

    def forward_test(self, bev_embed, outs_motion={}, outs_occflow={}, command=None, drivable_pred=None):
        sdc_traj_query = outs_motion['sdc_traj_query']
        sdc_track_query = outs_motion['sdc_track_query']
        bev_pos = outs_motion['bev_pos']
        occ_mask = outs_occflow.get('seg_out') if outs_occflow is not None else None
        
        outs_planning = self(bev_embed, occ_mask, bev_pos, sdc_traj_query, sdc_track_query, command, drivable_pred)
        return outs_planning

    def forward(self, 
                bev_embed, 
                occ_mask, 
                bev_pos, 
                sdc_traj_query, 
                sdc_track_query, 
                command,
                drivable_pred):
        """
        Forward pass for PlanningHeadSingleMode.

        Args:
            bev_embed (torch.Tensor): Bird's eye view feature embedding.
            occ_mask (torch.Tensor): Instance mask for occupancy.
            bev_pos (torch.Tensor): BEV position.
            sdc_traj_query (torch.Tensor): SDC trajectory query.
            sdc_track_query (torch.Tensor): SDC track query.
            command (int): Driving command.

        Returns:
            dict: A dictionary containing SDC trajectory and all SDC trajectories.
        """
        sdc_track_query = sdc_track_query.detach()
        sdc_traj_query = sdc_traj_query[-1]
        P = sdc_traj_query.shape[1]
        sdc_track_query = sdc_track_query[:, None].expand(-1,P,-1)
        
        
        command_context = torch.cat([sdc_traj_query, sdc_track_query], dim=-1).max(1)[0]
        command_logits = self.command_head(command_context)
        if self.predict_command:
            command_prob = torch.softmax(command_logits, dim=-1)
            navi_embed = command_prob @ self.navi_embed.weight
        else:
            command_tensor = self._command_to_tensor(command, sdc_traj_query.device, sdc_traj_query.shape[0])
            navi_embed = self.navi_embed.weight[command_tensor]
            command_prob = torch.softmax(command_logits, dim=-1)
        navi_embed = navi_embed[:, None].expand(-1, P, -1)
        plan_query = torch.cat([sdc_traj_query, sdc_track_query, navi_embed], dim=-1)

        plan_query = self.mlp_fuser(plan_query).max(1, keepdim=True)[0]   # expand, then fuse  # [1, 6, 768] -> [1, 1, 256]
        plan_query = rearrange(plan_query, 'b p c -> p b c')
        
        bev_pos = rearrange(bev_pos, 'b c h w -> (h w) b c')
        bev_feat = bev_embed +  bev_pos
        
        ##### Plugin adapter #####
        if self.with_adapter:
            bev_feat = rearrange(bev_feat, '(h w) b c -> b c h w', h=self.bev_h, w=self.bev_w)
            bev_feat = bev_feat + self.bev_adapter(bev_feat)  # residual connection
            bev_feat = rearrange(bev_feat, 'b c h w -> (h w) b c')
        ##########################
      
        pos_embed = self.pos_embed.weight
        plan_query = plan_query + pos_embed[None]  # [1, 1, 256]
        
        # plan_query: [1, 1, 256]
        # bev_feat: [40000, 1, 256]
        plan_query = self.attn_module(plan_query, bev_feat)   # [1, 1, 256]
        
        sdc_traj_all = self.reg_branch(plan_query).view((-1, self.planning_steps, 2))
        sdc_traj_all[...,:2] = torch.cumsum(sdc_traj_all[...,:2], dim=1)
        sdc_traj_all[0] = bivariate_gaussian_activation(sdc_traj_all[0])
        if self.use_col_optim and not self.training:
            # Post-process only when the perception masks are available.
            if occ_mask is not None:
                sdc_traj_all = self.collision_optimization(sdc_traj_all, occ_mask)
            if drivable_pred is not None:
                sdc_traj_all = self.drivable_optimization(sdc_traj_all, drivable_pred)
        
        return dict(
            sdc_traj=sdc_traj_all,
            sdc_traj_all=sdc_traj_all,
            command_logits=command_logits,
            command_prob=command_prob,
            pred_command=torch.argmax(command_logits, dim=-1),
        )

    def collision_optimization(self, sdc_traj_all, occ_mask):
        """
        Optimize SDC trajectory with occupancy instance mask.

        Args:
            sdc_traj_all (torch.Tensor): SDC trajectory tensor.
            occ_mask (torch.Tensor): Occupancy flow instance mask. 
        Returns:
            torch.Tensor: Optimized SDC trajectory tensor.
        """
        pos_xy_t = []
        valid_occupancy_num = 0
        
        if occ_mask.shape[2] == 1:
            occ_mask = occ_mask.squeeze(2)
        occ_horizon = occ_mask.shape[1]
        #assert occ_horizon == 5
        assert occ_horizon == (self.occ_n_future_only_occ+1)

        for t in range(self.planning_steps):
            cur_t = min(t+1, occ_horizon-1)
            pos_xy = torch.nonzero(occ_mask[0][cur_t], as_tuple=False)
            pos_xy = pos_xy[:, [1, 0]]
            pos_xy[:, 0] = (pos_xy[:, 0] - self.bev_h//2) * 0.5 + 0.25
            pos_xy[:, 1] = (pos_xy[:, 1] - self.bev_w//2) * 0.5 + 0.25

            # filter the occupancy in range
            keep_index = torch.sum((sdc_traj_all[0, t, :2][None, :] - pos_xy[:, :2])**2, axis=-1) < self.occ_filter_range**2
            pos_xy_t.append(pos_xy[keep_index].cpu().detach().numpy())
            valid_occupancy_num += torch.sum(keep_index>0)
        if valid_occupancy_num == 0:
            return sdc_traj_all
        
        col_optimizer = CollisionNonlinearOptimizer(self.planning_steps, 0.5, self.sigma, self.alpha_collision, pos_xy_t)
        col_optimizer.set_reference_trajectory(sdc_traj_all[0].cpu().detach().numpy())
        sol = col_optimizer.solve()
        sdc_traj_optim = np.stack([sol.value(col_optimizer.position_x), sol.value(col_optimizer.position_y)], axis=-1)
        return torch.tensor(sdc_traj_optim[None], device=sdc_traj_all.device, dtype=sdc_traj_all.dtype)
    
    def drivable_optimization(self, initial_trajectory, feasible_area):
        """
        Adjust the initial trajectory to ensure all points are within the feasible area using A* search.
        
        Args:
        initial_trajectory (torch.Tensor): Tensor of shape (6, 2) representing the initial trajectory.
        feasible_area (torch.Tensor): Tensor of shape (200, 200) representing the feasible area grid.
        
        Returns:
        torch.Tensor: Adjusted trajectory tensor of shape (6, 2).
        """
        def is_within_bounds(x, y, grid_size):
            return 0 <= x < grid_size and 0 <= y < grid_size
        
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def a_star_search(start, feasible_area):
            neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            close_set = set()
            came_from = {}
            gscore = {start: 0}
            fscore = {start: heuristic(start, start)}
            oheap = []
            
            heapq.heappush(oheap, (fscore[start], start))
            
            while oheap:
                current = heapq.heappop(oheap)[1]
                
                if feasible_area[current[0]][current[1]] == 1:
                    path = []
                    while current in came_from:
                        path.append(current)
                        current = came_from[current]
                    return path[::-1]
                
                close_set.add(current)
                for i, j in neighbors:
                    neighbor = current[0] + i, current[1] + j
                    tentative_g_score = gscore[current] + heuristic(current, neighbor)
                    if 0 <= neighbor[0] < feasible_area.shape[0]:
                        if 0 <= neighbor[1] < feasible_area.shape[1]:
                            if feasible_area[neighbor[0]][neighbor[1]] == 0:
                                continue
                        else:
                            continue
                    else:
                        continue
                    
                    if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                        continue
                    
                    if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                        came_from[neighbor] = current
                        gscore[neighbor] = tentative_g_score
                        fscore[neighbor] = tentative_g_score + heuristic(neighbor, start)
                        heapq.heappush(oheap, (fscore[neighbor], neighbor))
            
            return None
        
        def visualize_trajectory_and_feasible_area(initial_trajectory, feasible_area, filename):
            import cv2
            feasible_area_np = feasible_area.cpu().numpy() * 255
            feasible_area_color = cv2.cvtColor(feasible_area_np.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
            for i in range(initial_trajectory.size(0)):
                x, y = initial_trajectory[i].int().tolist()
                color = (0, 0, 255) if not is_feasible(x, y, feasible_area) else (0, 255, 0)
                cv2.circle(feasible_area_color, (y, x), 3, color, -1)
            filename = 'workspace/0805_debug_drivable/' + filename
            cv2.imwrite(filename, feasible_area_color)
        def real2bev(traj):
            real = [-51.2, 51.2]
            bev = [200, 200]
            bev_traj = traj.clone()
            bev_traj[:, 0] = (-traj[:, 1] + real[1]) / (real[1] - real[0]) * (bev[0] - 1)
            bev_traj[:, 1] = (traj[:, 0] + real[1]) / (real[1] - real[0]) * (bev[1] - 1)
            return torch.clamp(bev_traj, 0, bev[0]-1)
        def bev2real(traj):
            real = [-51.2, 51.2]
            bev = [200, 200]
            real_traj = traj.clone()
            real_traj[:, 0] = traj[:, 1] * ((real[1]-real[0])/bev[1]) + real[0]
            real_traj[:, 1] = -traj[:, 0] * ((real[1]-real[0])/bev[1]) + real[1]
            return real_traj
        def is_feasible(x, y, feasible_area):
            neighbors = [
                (0, 0), (0, 1), (0, -1), (1, 0), (1, 1), (1, -1), (-1, 0), (-1, 1), (-1, -1)
            ]
            feasible_count = 0
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if 0 <= nx < feasible_area.shape[1] and 0 <= ny < feasible_area.shape[0]:
                    if feasible_area[nx, ny] == 1:
                        feasible_count += 1
                else:
                    feasible_count += 1
            return feasible_count >= 2
        def is_consecutive(nums):
            for i in range(len(nums)-1):
                if nums[i] != nums[i+1] - 1:
                    return False
            return True
        
        ori_trajectory = initial_trajectory.clone()
        initial_trajectory = initial_trajectory[0]
        adjusted_trajectory = initial_trajectory.clone()
        grid_size = feasible_area.shape[0]
        initial_trajectory = real2bev(initial_trajectory)
        adjusted_trajectory = real2bev(adjusted_trajectory)

        non_feasible_points = []
        for i in range(initial_trajectory.size(0)):
            x, y = initial_trajectory[i].int().tolist()
            if not is_within_bounds(x, y, grid_size) or not is_feasible(x, y, feasible_area):
                non_feasible_points.append(i)
        
        # visualize_trajectory_and_feasible_area(initial_trajectory, feasible_area, str(self.num)+'_before.png')
        # if len(non_feasible_points) > 0:
        #     print('need optimization' + str(self.num))
        if len(non_feasible_points) == 0:
            return ori_trajectory
        if is_consecutive(non_feasible_points) and non_feasible_points[-1] == self.planning_steps-1:
            for i in non_feasible_points:
                adjusted_trajectory[i] = adjusted_trajectory[i-1]

        # print('after optimization')
        # visualize_trajectory_and_feasible_area(adjusted_trajectory, feasible_area, str(self.num)+'_after.png')

        adjusted_trajectory = bev2real(adjusted_trajectory)
        return adjusted_trajectory.unsqueeze(0)
    
    def loss(self, sdc_planning, sdc_planning_mask, outs_planning, future_gt_bbox=None, command=None):
        sdc_traj_all = outs_planning['sdc_traj_all'] # b, p, t, 5
        sdc_planning = self._batch_to_tensor(sdc_planning, sdc_traj_all.device)
        sdc_planning_mask = self._batch_to_tensor(sdc_planning_mask, sdc_traj_all.device)
        if sdc_planning.dim() == 3:
            sdc_planning = sdc_planning.unsqueeze(0)
        if sdc_planning_mask.dim() == 3:
            sdc_planning_mask = sdc_planning_mask.unsqueeze(0)
        loss_dict = dict()
        for i in range(len(self.loss_collision)):
            loss_collision = self.loss_collision[i](sdc_traj_all, sdc_planning[0, :, :self.planning_steps, :3], torch.any(sdc_planning_mask[0, :, :self.planning_steps], dim=-1), future_gt_bbox[0][1:self.planning_steps+1])
            loss_dict[f'loss_collision_{i}'] = loss_collision          
        loss_ade = self.loss_planning(sdc_traj_all, sdc_planning[0, :, :self.planning_steps, :2], torch.any(sdc_planning_mask[0, :, :self.planning_steps], dim=-1))
        loss_dict.update(dict(loss_ade=loss_ade))
        command_gt = self._command_to_tensor(command, sdc_traj_all.device, outs_planning['command_logits'].shape[0])
        if self.predict_command and command_gt is not None:
            loss_dict['loss_command'] = self.loss_command(outs_planning['command_logits'], command_gt)
        return loss_dict
