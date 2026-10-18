"""SB3 training-metric health thresholds, classification and fix recommendations.

Shared by the CustomTkinter app (rl_app.py) and the web GUI (webgui/). Pure data + functions.
"""

METRIC_INFO = {
    'approx_kl': {
        'good': (0.0, 0.05),
        'warn': (0.05, 0.15),
        'bad':  (0.15, float('inf')),
        'icon': '📐',
        'label': 'KL Divergence',
        'desc': 'Policy stability',
        'fix_high': lambda v: {
            'title': f'approx_kl = {v:.3f}',
            'severity': 'high' if v > 0.5 else 'med',
            'problem': 'Policy unstable — เปลี่ยนเร็วเกิน',
            'actions': [
                ('↓', 'learning_rate', '3e-4 → 1e-4', 'main'),
                ('↓', 'clip_range', '0.2 → 0.1', None),
                ('↓', 'window_size', 'current → 10', None),
            ],
        },
        'fix_low': lambda v: None,
    },
    'clip_fraction': {
        'good': (0.0, 0.2),
        'warn': (0.2, 0.3),
        'bad':  (0.3, 1.0),
        'icon': '✂️',
        'label': 'Clip Fraction',
        'desc': 'How often clipping triggers',
        'fix_high': lambda v: {
            'title': f'clip_fraction = {v:.3f}',
            'severity': 'high' if v > 0.5 else 'med',
            'problem': 'Policy พยายามกระโดด → โดน clip บ่อย',
            'actions': [
                ('↓', 'learning_rate', '3e-4 → 1e-4', 'main'),
                ('↑', 'n_steps', '2048 → 4096', None),
                ('⚠️', 'อย่าลด clip_range', 'จะแย่ลง!', 'warn'),
            ],
        },
        'fix_low': lambda v: None,
    },
    'explained_variance': {
        'good': (0.5, 1.0),
        'warn': (0.2, 0.5),
        'bad':  (-float('inf'), 0.2),
        'icon': '🎯',
        'label': 'Explained Variance',
        'desc': 'Critic accuracy',
        'fix_high': lambda v: None,
        'fix_low': lambda v: {
            'title': f'explained_variance = {v:.3f}',
            'severity': 'high' if v < 0 else 'med',
            'problem': 'Critic ทำนาย value แย่',
            'actions': [
                ('↑', 'Critic NN size', 'vf=[256,128,64]', 'main'),
                ('↑', 'gae_lambda', '0.95 → 0.97', None),
                ('↑', 'Training steps', '200k → 500k', None),
            ],
        },
    },
    'entropy_loss': {
        'good': (-2.0, -0.3),
        'warn': (-0.3, -0.1),
        'bad':  (-0.1, float('inf')),
        'icon': '🔍',
        'label': 'Entropy',
        'desc': 'Exploration level',
        'fix_high': lambda v: {
            'title': f'entropy_loss = {v:.3f}',
            'severity': 'med',
            'problem': 'Exploration หาย → policy collapse เร็ว',
            'actions': [
                ('↑', 'ent_coef', '0.01 → 0.05', 'main'),
                ('↓', 'learning_rate', '3e-4 → 1e-4', None),
                ('🔄', 'Restart', 'with different seed', None),
            ],
        },
        'fix_low': lambda v: {
            'title': f'entropy_loss = {v:.3f}',
            'severity': 'low',
            'problem': 'Exploration เยอะเกิน → ไม่ converge',
            'actions': [
                ('↓', 'ent_coef', '0.01 → 0.005', 'main'),
            ],
        },
    },
    'value_loss': {
        'good': (0.0, 0.01),
        'warn': (0.01, 0.1),
        'bad':  (0.1, float('inf')),
        'icon': '📉',
        'label': 'Value Loss',
        'desc': 'Critic training loss',
        'fix_high': lambda v: {
            'title': f'value_loss = {v:.4f}',
            'severity': 'high' if v > 1 else 'med',
            'problem': 'Critic เรียนไม่ทัน',
            'actions': [
                ('⚙️', 'Reward normalization', 'clip [-1, 1]', 'main'),
                ('↑', 'Critic NN size', 'vf=[256, 128]', None),
            ],
        },
        'fix_low': lambda v: None,
    },
    'ep_rew_mean': {
        'good': (0.5, float('inf')),
        'warn': (-0.1, 0.5),
        'bad':  (-float('inf'), -0.1),
        'icon': '💰',
        'label': 'Reward',
        'desc': 'Average episode reward',
        'fix_high': lambda v: None,
        'fix_low': lambda v: {
            'title': f'ep_rew_mean = {v:.2f}',
            'severity': 'high' if v < -0.5 else 'med',
            'problem': 'Agent กำลังขาดทุน',
            'actions': [
                ('⚙️', 'Reward function', 'ตรวจ logic', 'main'),
                ('↑', 'ent_coef', '0.01 → 0.05', None),
                ('⚙️', 'Reward shaping', 'penalty/bonus', None),
            ],
        },
    },
}


def classify_metric(name: str, value: float) -> str:
    """Returns 'good' | 'warn' | 'bad' | None"""
    info = METRIC_INFO.get(name)
    if not info:
        return None
    g_lo, g_hi = info['good']
    w_lo, w_hi = info['warn']
    if g_lo <= value <= g_hi:
        return 'good'
    elif w_lo <= value <= w_hi:
        return 'warn'
    else:
        return 'bad'


def get_recommendation(name: str, value: float):
    """Returns recommendation dict {title, severity, problem, actions} or None"""
    info = METRIC_INFO.get(name)
    if not info:
        return None
    status = classify_metric(name, value)
    if status == 'good':
        return None
    g_lo, g_hi = info['good']
    rec = None
    if value > g_hi and info.get('fix_high'):
        rec = info['fix_high'](value)
    elif value < g_lo and info.get('fix_low'):
        rec = info['fix_low'](value)
    if rec:
        rec['metric'] = name
        rec['icon'] = info.get('icon', '⚠️')
        rec['label'] = info.get('label', name)
    return rec
