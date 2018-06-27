library(RJSONIO)
library(devtools)
library(ROAuth)
library(httr)
library(httpuv)
install_github("tiagomendesdantas/Rspotify")
library(Rspotify)
library(assertthat)
library(bindr)
library(rlang)
library(randomForest)
library(caret)
library(e1071)
##
# Authentication (hardcoded)

keys <- spotifyOAuth("LoveDaSystem","81cc700dd4b14417bffd6f4fb52ac8c0","a34ddd6bd44a464f92f6ada8007ad2ac")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#g etPlaylist gets all playlists for a user
df<-getPlaylist("dekraus-us",offset=0,keys)

#The 26 playlist is the biggest
#The getPlaylistSongs function only fetches 100 at a time, but it does allow an offset
rock<-data.frame(getPlaylistSongs(df$ownerid[1],df$id[1],offset=0,keys))
jazz<-data.frame(getPlaylistSongs(df$ownerid[2],df$id[2],offset=0,keys))
folk<-data.frame(getPlaylistSongs(df$ownerid[3],df$id[3],offset=0,keys))
folk2<-data.frame(getPlaylistSongs(df$ownerid[3],df$id[3],offset=101,keys))
indie<-data.frame(getPlaylistSongs(df$ownerid[4],df$id[4],offset=0,keys))
hiphop<-data.frame(getPlaylistSongs(df$ownerid[5],df$id[5],offset=0,keys))
dance<-data.frame(getPlaylistSongs(df$ownerid[6],df$id[6],offset=0,keys))
classical<-data.frame(getPlaylistSongs(df$ownerid[7],df$id[7],offset=0,keys))
country<-data.frame(getPlaylistSongs(df$ownerid[8],df$id[8],offset=0,keys))

rock$genre<-as.factor("Rock")
jazz$genre<-as.factor("Jazz")
folk$genre<-as.factor("Folk")
folk2$genre<-as.factor("Folk")
indie$genre<-as.factor("Indie")
hiphop$genre<-as.factor("HipHop")
dance$genre<-as.factor("Dance")
classical$genre<-as.factor("Classical")
country$genre<-as.factor("Country")

playlist<-rbind(rock,jazz,folk,folk2,indie,hiphop,dance,classical,country,stringsAsFactors=FALSE)

#removing duplicate songs
anyDuplicated(playlist$id)
anyDuplicated(playlist$tracks)
playlist[playlist$tracks=="Stay Alive",]
playlist<-playlist[-183,]
anyDuplicated(playlist$albumId)

#create the df for analyzing
features <- data.frame(matrix(ncol=16,nrow=nrow(playlist)))
for(i in 1:100){
  features[i,] <- getFeatures(playlist[['id']][i], keys)
}
for(i in 101:200){
  features[i,] <- getFeatures(playlist[['id']][i], keys)
}
for(i in 201:300){
  features[i,] <- getFeatures(playlist[['id']][i], keys)
}
playlist<-playlist[-391,]
playlist<-playlist[-393,]
for(i in 301:400){
  features[i,] <- getFeatures(playlist[['id']][i], keys)
}
playlist<-playlist[-404,]
playlist<-playlist[-421,]
playlist<-playlist[-433,]
for(i in 401:nrow(playlist)){
  features[i,] <- getFeatures(playlist[['id']][i], keys)
}

features<-features[1:nrow(playlist),]


features$x17<-0
colnames(features)<-c('id','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence',
                      'tempo','duration_ms','time_signature','uri','analysis_url','genres')

features$genres<-playlist$genre

features[,15]<-NULL #run this part twice

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#SVM
set.seed(1234)
index <- sample(nrow(features), 0.75*nrow(features))
train <- features[index,]
test <- features[-index,]

Gamma <- 10^(-5:0)
Cost <- 10^(0:5)
tune <-  tune.svm(genres ~ .-id, data=train,
                  gamma=Gamma, cost=Cost)
summary(tune)

tuned.svm <- svm(genres ~.-id, data=train,
                 gamma=.001, cost=100,type="C-classification")
prediction <- predict(tuned.svm, test)
table(test$genres, prediction)
confusionMatrix(prediction,test$genres)
#Accuracy is .6048. Most error comes from rock category as well as indie
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

########Part 2
#randomforest

train$id<-as.factor(train$id)
test$id<-as.factor(test$id)
set.seed(1234)
fit.forest <- randomForest(genres ~ .-id, data=train, 
                           na.action=na.roughfix,
                           ntree=1000,
                           importance=TRUE)
pred<-predict(fit.forest,test)
confusionMatrix(pred,test$genres)
#58% accuracy. remove alternative and try again
#accuracy is 70% without alternative music. 
# "Alternative" is not well defined which would explain this misclassification

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#xgbBoost Trees
install.packages("xgboost")
library(xgboost)
#xggboost only does numeric
train$key=as.numeric(train$key)
test$key=as.numeric(test$key)
train$mode=as.numeric(train$mode)
test$mode=as.numeric(test$mode)
train$duration_ms=as.numeric(train$duration_ms)
test$duration_ms=as.numeric(test$duration_ms)
train$time_signature=as.numeric(train$time_signature)
test$time_signature=as.numeric(test$time_signature)
train$genres=as.numeric(train$genres)
test$genres=as.numeric(test$genres)

genre_matrix = model.matrix(train$genres)
train_matrix$genres <- train$genres

train$rock= ifelse(train$genres=="Rock", 1, 0)
train$indie=ifelse(train$genres=="Indie", 1, 0)
train$country=ifelse(train$genres=="Country", 1, 0)
train$hiphop=ifelse(train$genres=="HipHop", 1, 0)
train$country=ifelse(train$genres=="Country", 1, 0)
train$folk=ifelse(train$genres=="Folk", 1, 0)
train$jazz=ifelse(train$genres=="Jazz", 1, 0)
train$dance=ifelse(train$genres=="Dance", 1, 0)

test$rock= ifelse(test$genres=="Rock", 1, 0)
test$indie=ifelse(test$genres=="Indie", 1, 0)
test$country=ifelse(test$genres=="Country", 1, 0)
test$hiphop=ifelse(test$genres=="HipHop", 1, 0)
test$country=ifelse(test$genres=="Country", 1, 0)
test$folk=ifelse(test$genres=="Folk", 1, 0)
test$jazz=ifelse(test$genres=="Jazz", 1, 0)
test$dance=ifelse(test$genres=="Dance", 1, 0)

tc <- trainControl(method = "cv",
                   number = 10)

# random seed
set.seed(1234)
model2 <- train(rock+indie+country+hiphop+country+folk+jazz+dance ~ .-id , 
                data = train_matrix,
                method = "xgbTree",
                trControl = tc,
                tuneLength = 5,
                eta = 1,
                max_depth = 2)
model2
predict<-predict(model2,test)
confusionMatrix(predict,test$genres)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
