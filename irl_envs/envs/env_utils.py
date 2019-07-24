import os

ENV_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets')


def get_asset_xml(xml_name):
    return os.path.join(ENV_ASSET_DIR, xml_name)


def test_env(env, T=100):
    aspace = env.action_space
    env.reset()
    for t in range(T):
        o, r, done, infos = env.step(aspace.sample())
        print('---T=%d---' % t)
        print('rew:', r)
        print('obs:', o)
        env.render()
        if done:
            break

