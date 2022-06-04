import json
import time
import sys
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import csv
import tensorflow as tf
import tensorflow_hub as hub

list_final = list()

def connect2ES():
    # connect to ES on localhost on port 9200
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    if es.ping():
            print('Connected to ES!')
    else:
            print('Could not connect!')
            sys.exit()

    print("*********************************************************************************");
    return es

def keywordSearch(es, q):
    #Search by Keywords
    b={
            'query':{
                'match':{
                    "title":q
                }
            }
        }

    res= es.search(index='questions-index',body=b)
    
    list_1 = list()
    max_1 = -1.0
    min_1 = 100.0

    for hit in res['hits']['hits']:
        max_1 = max(float(hit['_score']), max_1)
        min_1 = min(float(hit['_score']), min_1)

    for hit in res['hits']['hits']:
        temp = float(hit['_score'])
        temp = (temp - min_1)/(max_1 - min_1)
        temp_1 = [temp, hit['_source']['title']]
        list_1.append(temp_1)
        #print(str(hit['_score']) + "\t" + hit['_source']['title'] )

    list_final = list_1
   #print("*********************************************************************************");

    return


# Search by Vec Similarity
def sentenceSimilaritybyNN(embed, es, sent):
    query_vector = tf.make_ndarray(tf.make_tensor_proto(embed([sent]))).tolist()[0]
    b = {"query" : {
                "script_score" : {
                    "query" : {
                        "match_all": {}
                    },
                    "script" : {
                        "source": "cosineSimilarity(params.query_vector, 'title_vector') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
             }
        }


    #print(json.dumps(b,indent=4))
    res= es.search(index='questions-index',body=b)

    res= es.search(index='questions-index',body=b)
    
    list_1 = list()
    max_1 = -1.0
    min_1 = 100.0
    
    for hit in res['hits']['hits']:
        max_1 = max(float(hit['_score']), max_1)
        min_1 = min(float(hit['_score']), min_1)

    for hit in res['hits']['hits']:
        temp = float(hit['_score'])
        temp = (temp - min_1)/(max_1 - min_1)
        temp_1 = [temp, hit['_source']['title']]
        list_1.append(temp_1)
        #print(str(hit['_score']) + "\t" + hit['_source']['title'] )

    list_final.extend(list_1)

   # print("*********************************************************************************");



if __name__=="__main__":

    es = connect2ES()
    embed = hub.load("./data/USE4/")
    
    while(1):
        query=input("Enter a Query:")

        start = time.time()
        if query=="END":
            break

        print("Query: " +query)
        keywordSearch(es, query)
        sentenceSimilaritybyNN(embed, es, query)
        list_final.sort()
        print("Hybrid Search Results")
        print("Similarity score" , "\t", "Questions")
        for i in range(10):
            print(list_final[i][0], "\t", list_final[i][1])
        end = time.time()
        print(end - start)
