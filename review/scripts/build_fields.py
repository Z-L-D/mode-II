# Builds F_<campaign>_<sub>.pkl = (gids, meta, X[G,T,L,3]=[g,gap,H], sigma) for complete generations.
import pickle; from common import *
for c,sub in [('ZIT','attn'),('ZIT','mlp'),('ZIB','attn'),('ZIB','mlp'),('MOCHI','attn'),('SD35M','attn'),('SD35L','attn'),('MOCHI1','attn')]:
    pickle.dump(fields(c,sub),open(f'F_{c}_{sub}.pkl','wb'))
