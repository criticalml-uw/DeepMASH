import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
from itertools import cycle


#import prediction data
pred_risk_death = pd.read_csv('pred_risk_death.csv',index_col=0)
pred_risk_transplant = pd.read_csv('pred_risk_transplant.csv',index_col=0)


#specifiy patient 
patient_1 = pd.DataFrame({
   'transplant': pred_risk_transplant.iloc[0][0:12],
   'death': pred_risk_death.iloc[0][0:12]
   })

marker2 = itertools.cycle(('o', 's')) 
lines2 = itertools.cycle(("-","--"))


x=range(0,12,1)


mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#006400','#c1272d']) 

fig,ax = plt.subplots(figsize=(10,10))

#plt.plot(x,patientt1,"-o")
ax.plot(x,patient_1['transplant'],
         linestyle=next(lines2),
         marker=next(marker2))

ax.plot(x,patient_1['death'],
         linestyle=next(lines2),
         marker=next(marker2))

ax.grid()
fig.suptitle('Surrogate risk of death and transplant using DeepNash for sample patient #3',fontsize=30)
ax.set_ylabel('Surrogate risks',fontsize=30)
ax.set_xlabel('Time in months',fontsize=30)

line_labels=["Transplant",'Death']
ax.legend(line_labels, loc ='lower center', borderaxespad=0.1, ncol=6, labelspacing=0.,  prop={'size': 22},
              bbox_to_anchor=(0.5, -0.2))

plt.xticks(visible=False)
plt.yticks(visible=False)

fig.set_size_inches(15,12, forward=True)
