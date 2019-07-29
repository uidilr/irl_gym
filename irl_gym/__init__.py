from gym.envs.registration import register

register(id='TwoDMaze-v0',
         entry_point='irl_gym.envs.twod_maze:TwoDMaze')
register(id='PointMazeLeft-v0', entry_point='irl_gym.envs.point_maze:PointMazeEnv',
         kwargs={'sparse_reward': False, 'direction': 0})
register(id='PointMazeRight-v0', entry_point='irl_gym.envs.point_maze:PointMazeEnv',
         kwargs={'sparse_reward': False, 'direction': 1})

# A modified ant which flips over less and learns faster via TRPO
register(id='CustomAnt-v0', entry_point='irl_gym.envs.ant_env:CustomAntEnv',
         kwargs={'gear': 30, 'disabled': False})
register(id='DisabledAnt-v0', entry_point='irl_gym.envs.ant_env:CustomAntEnv',
         kwargs={'gear': 30, 'disabled': True})

register(id='VisualPointMazeRight-v0', entry_point='irl_gym.envs.visual_pointmass:VisualPointMazeEnv',
         kwargs={'sparse_reward': False, 'direction': 1})
register(id='VisualPointMazeLeft-v0', entry_point='irl_gym.envs.visual_pointmass:VisualPointMazeEnv',
         kwargs={'sparse_reward': False, 'direction': 1})
