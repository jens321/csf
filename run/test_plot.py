import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# load np array
data = np.load('/anonymous/anonymous/metra-with-avalon/exp/ant_l2_penalty/sd000_s_55189870.0.1711468816_ant_metra/corr_m_2000.npy')

YLABELS = [
    'torso z coord',
    'torso x orient',
    'torso y orient',
    'torso z orient',
    'torso w orient',
    'angle torso, first link front left',
    'angle two links, front left',
    'angle torso, first link front right',
    'angle two links, front right',
    'angle torso, first link back left',
    'angle two links, back left',
    'angle torso, first link back right',
    'angle two links, back right',
    'torso x coord vel',
    'torso y coord vel',
    'torso z coord vel',
    'torso x coord ang vel',
    'torso y coord ang vel',
    'torso z coord ang vel',
    'angle torso, front left link, av',
    'angle front left links, av',
    'angle torso, front right link, av',
    'angle front right links, av',
    'angle torso, back left link, av',
    'angle back left links, av',
    'angle torso, back right link, av',
    'angle back right links, av',
    'torso x coord',
    'torso y coord',
]

# plot heatmap with set max and min
sns.heatmap(data, cmap='coolwarm', center=0)
plt.yticks(ticks=list(map(lambda x: x + 0.5, range(29))), labels=YLABELS, rotation=0)


plt.xlabel('Z')
plt.ylabel('Observations')
plt.tight_layout()

plt.savefig('ant_metra_corr_m.pdf')