feat_db_path = './DB/feat-db.csv'
model_path = './generator/output/FaceGen.RaFD.model.d6.adam.iter500.h5'

from Pipeline import Pipeline

p = Pipeline(feat_db_path, model_path)
p.processSequence('./in/P1L_S1_C1/', './out/deid-test/', None, False, True, False)
