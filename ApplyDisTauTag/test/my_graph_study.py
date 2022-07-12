import tensorflow as tf

file_ = '/afs/desy.de/user/m/mykytaua/nfscms/softDeepTau/RecoML/DisTauTag/TauMLTools/ApplyDisTauTag/data/graph.pb'

with tf.io.gfile.GFile(file_,'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

print(dir(graph_def))

# print(graph_def.DESCRIPTOR())

print(graph_def.ListFields())
