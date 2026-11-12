"""Custom colormaps of this project, registered with matplotlib when this module is imported (replay.simdata
imports it), so their names work wherever a colormap is chosen, cut versions 'name-T' included. Reloading the
module re-registers them, so an edited map replaces the old one in a running kernel.

octarine_tuned (and octarine_tuned_r): cmyt.octarine, whole range, lightness remapped to J' 20 -> 74
(CAM02-UCS, linear), hue and colourfulness unchanged; chosen with notebooks/cmap_tuner.ipynb for the
stellar-mass orbits (Comp01b). Stored as its 256 colours so it never changes with library versions;
recreate with replay.cmaptune:
    retune('cmyt.octarine', window=(0, 1), J=(20, 74), gamma=1, hue=0, chroma=1)

iceburn-custom (and iceburn-custom_r): cmr.iceburn with a narrower black centre (diverging: blue,
black at 0.5, orange), for ratios around 1 (Comp02b); cut versions work as for any name, e.g.
'iceburn-custom-0.9' (0 -> 0.9 of it). Recreate with replay.cmaptune (tune it in the tuner's
diverging section):
    squeeze('cmr.iceburn', middle=0.5)

ghostlight_tuned (and ghostlight_tuned_r): cmr.ghostlight, whole range, lightness remapped to J' 20 -> 74
(as octarine_tuned) and its colourfulness raised x1.8 at fixed lightness and hue (colours beyond the
screen's gamut brought back to its edge, not clipped); chosen for the dark-matter-mass orbits (Comp01c,
Comp01d's bottom panel). Recreate with replay.cmaptune:
    boost(retune('cmr.ghostlight', window=(0, 1), J=(20, 74), gamma=1, hue=0, chroma=1), 1.8)
"""
from matplotlib import colormaps
from matplotlib.colors import ListedColormap

OCTARINE_TUNED = [
    '#0d2c58', '#0d2d57', '#0e2e56', '#0e2f55', '#0e2f54', '#0f3053', '#0f3152', '#103251',
    '#103351', '#113350', '#11344f', '#12354e', '#13354e', '#13364d', '#14374d', '#14374c',
    '#15384b', '#15394b', '#16394a', '#173a4a', '#173a49', '#183b49', '#183c49', '#193c48',
    '#193d48', '#1a3e47', '#1a3e47', '#1b3f47', '#1b3f46', '#1b4046', '#1c4145', '#1c4145',
    '#1d4245', '#1d4244', '#1d4344', '#1e4444', '#1e4443', '#1e4543', '#1f4643', '#1f4642',
    '#1f4742', '#1f4741', '#204841', '#204940', '#204940', '#214a3f', '#214b3f', '#214b3e',
    '#224c3e', '#224c3d', '#234d3d', '#234e3c', '#244e3b', '#244f3b', '#254f3a', '#255039',
    '#265038', '#275138', '#285237', '#285236', '#295335', '#2a5334', '#2b5433', '#2c5432',
    '#2e5531', '#2f5530', '#30552f', '#32562e', '#33562d', '#34572c', '#36572b', '#37572a',
    '#395829', '#3b5828', '#3c5827', '#3e5826', '#405925', '#425924', '#435923', '#455922',
    '#475921', '#495a20', '#4b5a1f', '#4c5a1f', '#4e5a1e', '#505a1d', '#525a1c', '#545a1b',
    '#555a1b', '#575a1a', '#595a19', '#5b5a18', '#5d5a18', '#5e5b17', '#605b17', '#625b16',
    '#645b15', '#655a15', '#675a14', '#695a14', '#6b5a13', '#6c5a13', '#6e5a13', '#705a12',
    '#715a12', '#735a11', '#755a11', '#765a11', '#785a11', '#7a5a10', '#7b5a10', '#7d5910',
    '#7f5910', '#805910', '#82590f', '#84590f', '#85590f', '#87580f', '#89580f', '#8a580f',
    '#8c580f', '#8e580f', '#8f570f', '#91570f', '#92570f', '#94570f', '#965610', '#975610',
    '#995610', '#9a5610', '#9c5510', '#9e5511', '#9f5511', '#a15411', '#a35411', '#a45312',
    '#a65312', '#a75312', '#a95213', '#ab5213', '#ac5113', '#ae5114', '#af5014', '#b15015',
    '#b34f15', '#b44f16', '#b64e16', '#b84e17', '#b94d17', '#bb4c18', '#bc4c18', '#be4b19',
    '#c04b1a', '#c14a1a', '#c3491b', '#c5481c', '#c6481d', '#c8471e', '#c9461e', '#cb451f',
    '#cd4420', '#ce4421', '#d04323', '#d14224', '#d34125', '#d44026', '#d63f28', '#d73f29',
    '#d93e2b', '#da3d2d', '#db3c2f', '#dc3c31', '#de3b33', '#df3b35', '#df3b38', '#e03b3a',
    '#e13c3d', '#e13c40', '#e23d42', '#e23e45', '#e23f48', '#e3404b', '#e3414d', '#e34350',
    '#e34453', '#e34655', '#e34758', '#e3485a', '#e34a5d', '#e24b5f', '#e24d62', '#e24e64',
    '#e25067', '#e25169', '#e1536c', '#e1546e', '#e15670', '#e15773', '#e05975', '#e05a77',
    '#e05b7a', '#df5d7c', '#df5e7e', '#df6080', '#de6183', '#de6285', '#de6487', '#dd6589',
    '#dd678b', '#dc688d', '#dc698f', '#dc6a92', '#db6c94', '#db6d96', '#da6e98', '#da709a',
    '#da719c', '#d9729e', '#d973a0', '#d875a2', '#d876a4', '#d877a6', '#d778a7', '#d779a9',
    '#d67bab', '#d67cad', '#d67daf', '#d57eb1', '#d57fb3', '#d480b4', '#d481b6', '#d482b8',
    '#d383ba', '#d384bc', '#d386bd', '#d287bf', '#d288c1', '#d289c3', '#d18ac5', '#d18bc6',
    '#d18cc8', '#d08cca', '#d08dcc', '#d08ece', '#d08fcf', '#d090d1', '#cf91d3', '#cf92d5',
    '#cf93d7', '#cf93d9', '#cf94db', '#cf95dd', '#d095df', '#d195e1', '#d395e2', '#d595e2',
]

ICEBURN_CUSTOM = [
    '#94f1f3', '#92f0f3', '#91eff2', '#8feef2', '#8eedf2', '#8decf1', '#8beaf1', '#8ae9f0',
    '#88e8f0', '#87e7f0', '#85e6ef', '#84e5ef', '#83e3ee', '#81e2ee', '#80e1ee', '#7ee0ed',
    '#7ddfed', '#7bdeec', '#7adcec', '#78dbec', '#77daeb', '#75d9eb', '#73d8eb', '#72d6ea',
    '#70d5ea', '#6fd4ea', '#6dd3e9', '#6cd1e9', '#6ad0e9', '#68cfe8', '#67cee8', '#65cde8',
    '#63cbe7', '#62cae7', '#60c9e7', '#5ec8e6', '#5cc6e6', '#5bc5e6', '#59c4e6', '#57c3e5',
    '#55c1e5', '#54c0e5', '#52bfe5', '#50bde4', '#4ebce4', '#4cbbe4', '#4ab9e4', '#49b8e3',
    '#47b7e3', '#45b5e3', '#43b4e3', '#41b3e3', '#3fb1e2', '#3db0e2', '#3bafe2', '#39ade2',
    '#37ace2', '#35aae2', '#33a9e1', '#31a7e1', '#2fa6e1', '#2da4e1', '#2ba3e1', '#29a1e1',
    '#27a0e1', '#259ee0', '#239de0', '#219be0', '#1f9ae0', '#1d98e0', '#1c96e0', '#1b95e0',
    '#1993df', '#1891df', '#188fdf', '#188edf', '#188cde', '#188ade', '#1988de', '#1a86dd',
    '#1b84dd', '#1d82dc', '#1f80db', '#217eda', '#237cd9', '#267ad8', '#2878d7', '#2a76d5',
    '#2d74d3', '#2f72d1', '#3170cf', '#336dcc', '#356bc9', '#3769c6', '#3867c3', '#3965bf',
    '#3a63bc', '#3b62b8', '#3c60b4', '#3d5eb0', '#3d5cac', '#3d5aa8', '#3d58a4', '#3d56a0',
    '#3d549b', '#3d5297', '#3c5093', '#3c4e8e', '#3b4c89', '#3a4a85', '#394880', '#38467b',
    '#374376', '#364171', '#353f6c', '#333c67', '#323a62', '#30375c', '#2e3457', '#2c3151',
    '#292e4b', '#272b44', '#24273e', '#202336', '#1c1e2e', '#171925', '#11121b', '#07070b',
    '#0c0607', '#1b0e10', '#261316', '#2f171b', '#361a1e', '#3d1d21', '#431f23', '#492125',
    '#4e2327', '#532528', '#582729', '#5d292a', '#612a2b', '#652c2b', '#692d2c', '#6d2e2c',
    '#71302c', '#75312c', '#78322c', '#7c342c', '#7f352c', '#82362c', '#86382b', '#89392b',
    '#8b3b2a', '#8e3c2a', '#913d29', '#943f29', '#964028', '#994228', '#9b4327', '#9e4526',
    '#a04626', '#a24825', '#a44924', '#a64b23', '#a84d23', '#aa4e22', '#ac5021', '#ae5121',
    '#b05320', '#b2551f', '#b3561f', '#b5581e', '#b65a1d', '#b85b1d', '#ba5d1c', '#bb5f1b',
    '#bd601b', '#be621a', '#bf641a', '#c16619', '#c26719', '#c36918', '#c56b18', '#c66c17',
    '#c76e17', '#c87017', '#c97116', '#ca7316', '#cc7516', '#cd7615', '#ce7815', '#cf7a15',
    '#d07b15', '#d17d15', '#d27f15', '#d38015', '#d48215', '#d48415', '#d58515', '#d68715',
    '#d78916', '#d88a16', '#d98c16', '#da8e17', '#da8f17', '#db9117', '#dc9218', '#dd9418',
    '#dd9619', '#de9719', '#df991a', '#e09b1b', '#e09c1b', '#e19e1c', '#e29f1d', '#e2a11d',
    '#e3a31e', '#e3a41f', '#e4a620', '#e5a721', '#e5a921', '#e6aa22', '#e6ac23', '#e7ae24',
    '#e8af25', '#e8b126', '#e9b227', '#e9b428', '#eab529', '#eab72a', '#ebb92b', '#ebba2c',
    '#ecbc2c', '#ecbd2d', '#edbf2e', '#edc02f', '#eec230', '#eec332', '#eec533', '#efc634',
    '#efc835', '#f0c936', '#f0cb37', '#f1cc38', '#f1ce39', '#f1cf3a', '#f2d13b', '#f2d23c',
    '#f3d43d', '#f3d63e', '#f3d73f', '#f4d940', '#f4da41', '#f4dc42', '#f5dd44', '#f5de45',
]

GHOSTLIGHT_TUNED = [
    '#2c2b2b', '#2c2b2c', '#2d2c2d', '#2d2c2e', '#2d2d30', '#2e2d31', '#2e2e33', '#2e2e34',
    '#2e2f36', '#2f2f37', '#2f2f39', '#2f303a', '#30303c', '#30313e', '#313140', '#313141',
    '#313243', '#323245', '#323247', '#333349', '#33334b', '#34334d', '#343350', '#353352',
    '#353454', '#363456', '#373459', '#38345b', '#38345d', '#393460', '#3a3462', '#3b3465',
    '#3c3368', '#3d336a', '#3e336d', '#3f3370', '#403273', '#413276', '#433179', '#44307c',
    '#45307f', '#472f82', '#482e85', '#4a2d88', '#4c2b8c', '#4e2a8f', '#4f2892', '#512795',
    '#532499', '#55229c', '#57209f', '#591ca3', '#5c19a6', '#5e15a9', '#600fac', '#6208af',
    '#6400b2', '#6600b1', '#6700b1', '#6800b1', '#6a00b0', '#6b00b0', '#6c00af', '#6e00af',
    '#6f00ae', '#7100ad', '#7200ad', '#7300ac', '#7500ab', '#7600ab', '#7700aa', '#7900a9',
    '#7a00a8', '#7c00a7', '#7d00a7', '#7e00a6', '#8000a5', '#8100a4', '#8200a4', '#8300a3',
    '#8500a2', '#8600a1', '#8700a1', '#88059f', '#880d9c', '#88139a', '#881897', '#881c95',
    '#882093', '#882390', '#88268e', '#88298c', '#882b8a', '#882e88', '#883086', '#883285',
    '#883483', '#883681', '#883880', '#87397e', '#873b7c', '#873d7b', '#873e7a', '#874078',
    '#874177', '#874376', '#874474', '#874673', '#874772', '#874871', '#874a70', '#864b6f',
    '#864c6e', '#864d6d', '#864e6c', '#86506b', '#86516a', '#865269', '#865368', '#865468',
    '#865567', '#865666', '#865765', '#865865', '#865964', '#865a63', '#865b62', '#865c61',
    '#865d61', '#865e60', '#865f5f', '#86605e', '#86615e', '#86625d', '#86635c', '#86645b',
    '#86655a', '#866659', '#866759', '#866758', '#866857', '#866956', '#856a55', '#856b54',
    '#856c53', '#856d52', '#856e52', '#846f51', '#847050', '#84714f', '#84724e', '#83734d',
    '#83744c', '#83754c', '#83764b', '#82774a', '#827849', '#827948', '#817a47', '#817b46',
    '#807c45', '#807d44', '#807e43', '#7f7f41', '#7f8040', '#7e813f', '#7e823e', '#7d833c',
    '#7d843b', '#7c853a', '#7b8638', '#7b8737', '#7a8835', '#7a8933', '#798a32', '#788b30',
    '#778c2e', '#778d2c', '#768e2a', '#758f28', '#749025', '#739223', '#729320', '#71941d',
    '#709519', '#6f9615', '#6e9711', '#6d980b', '#6c9903', '#6b9a00', '#6a9b00', '#6a9c00',
    '#699d00', '#699d00', '#689e00', '#689f00', '#67a000', '#67a100', '#66a100', '#66a200',
    '#66a300', '#66a400', '#65a400', '#65a500', '#65a600', '#65a600', '#65a700', '#65a800',
    '#66a800', '#66a900', '#66aa00', '#67aa00', '#68ab00', '#68ab00', '#69ac00', '#6bac00',
    '#6cad00', '#6dad00', '#6fae00', '#71ae00', '#73ae00', '#75ae00', '#78ae00', '#7aaf00',
    '#7daf00', '#7faf00', '#81af00', '#84af00', '#86af00', '#89af00', '#8baf00', '#8daf00',
    '#8faf00', '#92af00', '#94b000', '#96b000', '#98b000', '#9ab000', '#9cb000', '#9eb000',
    '#a1b000', '#a3b000', '#a5b000', '#a7b000', '#a9b100', '#aab100', '#acb100', '#aeb100',
    '#b0b100', '#b2b100', '#b4b100', '#b6b100', '#b8b100', '#bab100', '#bbb100', '#bdb200',
]

CUSTOM = {'octarine_tuned': OCTARINE_TUNED, 'iceburn-custom': ICEBURN_CUSTOM, 'ghostlight_tuned': GHOSTLIGHT_TUNED}


def register():
    """Register (or replace) every custom map and its reverse (_r) under its name."""
    import warnings
    for name, colours in CUSTOM.items():
        cm = ListedColormap(colours, name=name)
        for c in (cm, cm.reversed()):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Overwriting the cmap')
                colormaps.register(c, force=True)


register()
