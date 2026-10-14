import json

with open(r'D:\reinforcement learning\asl-github-repo\rl-asl-4pld.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Unit test assert
        if 'ATT_MIN, ATT_MAX = 0.5, 3.0' in source and 'assert' not in source:
            source = source.replace('ATT_MIN, ATT_MAX = 0.5, 3.0', 'ATT_MIN, ATT_MAX = 0.5, 3.0\nassert ATT_MAX <= 5.0, "ATT_MAX must be in seconds"')
            cell['source'] = [line + '\n' for line in source.split('\n')]
            
        # Environment fixes
        if 'class ASLEnvironment:' in source:
            # 1. Fix _compute_param_error
            old_err = '''    def _compute_param_error(self):
        return torch.sqrt(
            (self._norm(self.cbf_true, CBF_MIN, CBF_MAX) - self._norm(self.cbf_est, CBF_MIN, CBF_MAX)) ** 2 +
            (self._norm(self.att_true, ATT_MIN, ATT_MAX) - self._norm(self.att_est, ATT_MIN, ATT_MAX)) ** 2)'''
            new_err = '''    def _compute_param_error(self):
        cbf_err = torch.abs(self.cbf_true - self.cbf_est) / (CBF_MAX - CBF_MIN)
        att_err = torch.abs(self.att_true - self.att_est) / (ATT_MAX - ATT_MIN)
        return cbf_err + att_err'''
            source = source.replace(old_err, new_err)
            
            # 2. Fix reward calculation
            old_rew = 'reward = (self.prev_param_err - cur_param_err) * 10.0 + self.CCC_WEIGHT * self._compute_ccc() - 0.02'
            new_rew = 'reward = torch.clamp((self.prev_param_err - cur_param_err) * 10.0, -5.0, 5.0)'
            source = source.replace(old_rew, new_rew)
            
            # 3. Fix done condition
            old_done = 'done = cur_param_err < 0.01'
            new_done = 'done = torch.zeros_like(timeout, dtype=torch.bool) # Fixed horizon MDP'
            source = source.replace(old_done, new_done)
            
            cell['source'] = [line + '\n' for line in source.split('\n')]
            cell['source'] = [s for s in cell['source'] if s != '\n']

        # Evaluation Cells Fix
        if 'att_t_sec = att_t / 1000.0' in source:
            source = source.replace('att_t_sec = att_t / 1000.0', 'att_t_sec = att_t  # FIX: Already in seconds')
            source = source.replace('att_e_sec = att_e / 1000.0', 'att_e_sec = att_e  # FIX: Already in seconds')
            cell['source'] = [line + '\n' for line in source.split('\n')]
            cell['source'] = [s for s in cell['source'] if s != '\n']
            
        if 'att_t, att_e = env.att_true / 1000.0, env.att_est / 1000.0' in source:
            source = source.replace('att_t, att_e = env.att_true / 1000.0, env.att_est / 1000.0', 'att_t, att_e = env.att_true, env.att_est  # FIX: Already in seconds')
            cell['source'] = [line + '\n' for line in source.split('\n')]
            cell['source'] = [s for s in cell['source'] if s != '\n']

with open(r'D:\reinforcement learning\asl-github-repo\rl-asl-4pld.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Notebook fixed.')

