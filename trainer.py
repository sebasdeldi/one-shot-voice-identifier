from models import *
from utils import *


def average_emb(X_predicts,Y_predicts):
  average_embeddings = {}
  count_embeddings = {}
  for i in range(len(X_predicts)):
      if Y_predicts[i] not in average_embeddings:
          average_embeddings[Y_predicts[i]] = X_predicts[i]
          count_embeddings[Y_predicts[i]] = 1
      else:
          average_embeddings[Y_predicts[i]] += X_predicts[i]
          count_embeddings[Y_predicts[i]] += 1

  for key, value in average_embeddings.items():
      average_embeddings[key] = average_embeddings[key]/count_embeddings[key]
  return average_embeddings

  
def get_acurracy(X_predicts,Y_train):
  average_embeddings = average_emb(X_predicts,Y_train)
  count = 0
  bad_count = 0
  for i in range(len(X_predicts)):
    best_result = -1
    best_val = 0
    for key, value in average_embeddings.items():
        result = cosine_similarity(X_predicts[i],value)
        #print (key, Y_test[i],result)
        if result > best_result:
            best_result = result
            best_val = key
            
    if (best_val == Y_train[i]):
        count+=1
    else:
        bad_count+=1

  return (count/(count+bad_count))


def make_matrix(X_train,Y_train,average_model):
    print('making matrix')
    X_predicts = average_model.predict(X_train,batch_size = 16,verbose = 1)
    accuracy = get_acurracy(X_predicts,Y_train)
    matrix = np.zeros((len(X_train[0]),len(X_train[0])))
    for i in range(len(X_predicts)):
        matrix[i][i] = 1.0
        for j in range(i+1,len(X_predicts)):
            matrix[i][j] = cosine_similarity(X_predicts[i],X_predicts[j])
            matrix[j][i] = matrix[i][j]
    return accuracy,matrix

def make_worse_triplets(matrix,X_train,Y_train):
    anchors = np.zeros((5,len(X_train[0]),101,318))
    positives = np.zeros((5,len(X_train[0]),101,318))
    negatives = np.zeros((5,len(X_train[0]),101,318))
    for x in range(len(matrix)):
        lowest = 1.5
        highest = -1.5
        anchor = X_train[:,x:x+1]
        for y in range(len(matrix)):
            if matrix[x][y] < lowest and (Y_train[x] == Y_train[y]):
                lowest = matrix[x][y]
                positive = y
            elif matrix[x][y] > highest and Y_train[x] != Y_train[y]:
                highest = matrix[x][y]
                negative = y
        anchors[:,x,:,:] = X_train[:,x,:,:]
        positives[:,x,:,:] = X_train[:,positive,:,:]
        negatives[:,x,:,:] = X_train[:,negative,:,:]
        #print (x,'/',len(matrix))
    return anchors,positives,negatives

def train_cicle(X_train,Y_train,base_model,speech_model):
    epochs = 50
    batch_size = 32
    Y_dummy = np.empty((batch_size, 3))
    messages= []
    accuracies = []
    for i in range(epochs):
        print ("epoch",i)
        accuracy,matrix = make_matrix(X_train.tolist(),Y_train,base_model)
        print ("made matrix")
        print("acurracy",accuracy)
        accuracies += [accuracy]
        anchors, positives, negatives = make_worse_triplets(matrix,X_train,Y_train)
        start_idx = 0
        mean_loss = 0
        while (start_idx + batch_size < len(anchors[0])):
            batch_x = anchors[:,start_idx:start_idx + batch_size].tolist()+positives[:,start_idx:start_idx + batch_size].tolist()+negatives[:,start_idx:start_idx + batch_size].tolist()
            message = speech_model.train_on_batch(x = batch_x, y = Y_dummy)
            mean_loss+= message[0]
            messages+= [message]
            if ((start_idx/batch_size)%3 == 0):
                print ("start_idx:",start_idx, ",loss:",message)
            start_idx+=batch_size
        if (len(anchors[0])%batch_size != 0):
            batch_x = anchors[:,-batch_size:].tolist()+positives[:,-batch_size:].tolist()+negatives[:,-batch_size:].tolist()
            speech_model.train_on_batch(x = batch_x, y = Y_dummy)
        print("average_loss:",(mean_loss/(len(anchors[0])/batch_size+1)))
    return messages, accuracies

def main():
    Tx = 318
    n_freq = 101

    base_model_created = base_model(input_shape = (n_freq, Tx))
    average_model_created = five_average_model(input_shape = (n_freq, Tx), base_model = base_model_created)
    speech_model_created = speech_model(input_shape = (n_freq, Tx), average_model = average_model_created) 
    speech_model_created.compile(loss=triplet_loss, optimizer='adam', metrics=["accuracy"])

    X,Y = process_dataset()

    X_train, Y_train, X_test, Y_test = split_dataset(X,Y,0.8)

    messages,acurracies = train_cicle(X_train,Y_train,base_model_created,speech_model_created)
    
    np.save('messages',messages)
    np.save('acc',acurracies)

    X_test_predict = average_model_created.predict(X_test.tolist(),batch_size = 8,verbose = 1)
    
    acc = get_acurracy(X_test_predict,Y_train)
    
    print (acc)


