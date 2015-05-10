# MLProject
## With TFIDF:

Decision Tree:  71% 

Random Froest: 83%

SVM: 81% --> very slow at training (585.426215887)
comment:yes, and it even takes much longer time to find the optimal penalty parameter 
updated: SVM 83.5% with Lambda=5 (Run time: 584.36698103)


Logistics: 84%

KNN: 66%  --> very slow at testing

Adaboost: 74%

Linear Regressin: has bug   comment: I dont see how linear regression could be used for classification ??
We could use it by mapping score >0.5 --> 1 and <0.5 --> 0

## Without TFIDF:

### comment: the TFIDF doesn't imporve the performance 

Decision Tree:  70% 

Random Froest: 82%

SVM: 

Logistics: 83%

Logistics with L2 :"testAccuracy": 0.8376196172248804,


KNN: 

Adaboost: 

Linear Regressin: has bug   comment: I dont see how linear regression could be used for classification ??
