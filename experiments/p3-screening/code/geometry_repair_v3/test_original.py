import ast,json,unittest
from pathlib import Path
import numpy as np

class OriginalRegression(unittest.TestCase):
 def test_each_atom_pair_counted_once(self):
  tree=ast.parse(Path(__file__).with_name('original_extract.py').read_text())
  loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and [getattr(t,'id',None) for t in n.target.elts]==['a','ae'])
  scope={'np':np,'le':['C'],'elems':['N','O'],'dist':np.array([[1.,4.]]),'bins':np.array([0,2,3,4,5,6,8,10,np.inf]),'keys':[],'vals':[]}
  exec(compile(ast.Module(body=[loop],type_ignores=[]),'original-loop','exec'),scope)
  self.assertEqual(sum(scope['vals']),2,'Each real atom pair must contribute once')
 def test_split_join(self):
  sp=json.loads(Path('/workspace/ayb/experiments/dfe-unified-p3/refined_split.json').read_text())
  old={}
  for k,v in sp.items():
   if isinstance(v,list):
    for code in v: old[str(code)]=k
   elif isinstance(v,dict):
    for code,s in v.items(): old[str(code)]=str(s)
  self.assertEqual(old.get('10gs'),next(s for s,cs in sp['split'].items() if '10gs' in cs))

if __name__=='__main__':unittest.main()
