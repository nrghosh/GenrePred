
# Problem Statement
What even are genres? Genres are a special method that we use to classify music. While some types of classifications are extremely subjective, music has evolved to a point where we can draw lines, albeit blurry ones, in the sand, and make statements about two songs, asserting them to different categories. However, how can classifying every song by hand be practical? Manually sorting tracks into categories is both labor-intensive and time consuming, while sorting tracks into their respective genres can be a subjective exercise, vulnerable to personal biases and skewed perspectives. In our final project, we attempt to train a few types of models to classify songs to certain genres. 

# Data Description
```{r libraries, echo=TRUE, results='hide',message=FALSE, eval=FALSE}
# Load appropriate libraries
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
library(purrr)

# Spotify Authentication
keys <- spotifyOAuth("LoveDaSystem","81cc700dd4b14417bffd6f4fb52ac8c0","a34ddd6bd44a464f92f6ada8007ad2ac")

#getPlaylist gets all playlists for a user (Dennis, in this case)
df<-getPlaylist("dekraus-us",offset=0,keys)
```

The data we used for this project was obtained from Spotify, by using the Rspotify package. We used the Spotify API to import a playlist into R, and took advantage of the fact that the Spotify API allows us to get variables (attributes) of each song, including Artist, Album, Popularity (0-100), Danceability, Energy (0-1), Key, Loudness, Mode, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo, Duration (in ms), and Time Signature.

# Data Preprocessing
```{r dfmanagement, echo=TRUE, results='hide',message=FALSE, eval=FALSE}
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

# factorize the genres
rock$genre<-as.factor("Rock")
jazz$genre<-as.factor("Jazz")
folk$genre<-as.factor("Folk")
folk2$genre<-as.factor("Folk")
indie$genre<-as.factor("Indie")
hiphop$genre<-as.factor("HipHop")
dance$genre<-as.factor("Dance")
classical$genre<-as.factor("Classical")
country$genre<-as.factor("Country")

# Put it all together!
playlist<-rbind(rock,jazz,folk,folk2,indie,hiphop,dance,classical,country,stringsAsFactors=FALSE)

# Remove any duplicate songs
anyDuplicated(playlist$id)
anyDuplicated(playlist$tracks)
playlist[playlist$tracks=="Stay Alive",]
playlist<-playlist[-183,]
```

```{r makedf, echo=TRUE, results='hide',message=FALSE, eval=FALSE}
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

#run line twice
features[,15]<-NULL
features[,15]<-NULL
```
We already had the "genre" variable for each song from Spotify, so we converted that variable to a factor for each instance. We then created a dataframe called features for us to analyze and play with the data. We also ensured that we removed duplicate songs, wherever present. 

# ML Approach

### Support Vector Machine
In order to create, train and test our SVM model, we divided the dataset into a training and testing subset. A Support Vector Machine is a method for classifying data. It falls under the category of supervised learning, which means that the model estimates some classifying function by looking at labeled training data (in this case, examples of songs that have been categorized into specific genres.) It works by representing aspects of the data in physical space, and creating imaginary lines (or vectors) in order to accurately separate the data into categories.
```{r svm, echo=TRUE, results='hide',message=FALSE, eval=FALSE}
#SVM
set.seed(123)
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
# Accuracy is .6935. Majority of the error comes from the rock category as well as indie.
```

### Random Forest
We then followed a similar process for building a Random Forest model. A random forest model is an "ensemble learning" technique, which means that we build up a lot of full grown decision trees (low bias, high variance) in parallel, and then aggregate them into one model. A decision tree is a model that represents possible outcomes along with their probabilities. By building a lot of these models and taking the average, we can predict the genre of a song to a certain degree of certainty. 
```{r randomforest, echo=TRUE, results='hide',message=FALSE, eval=FALSE}
# Random Forest

train$id<-as.factor(train$id)
test$id<-as.factor(test$id)
set.seed(123)
fit.forest <- randomForest(genres ~ .-id, data=train, 
                           na.action=na.roughfix,
                           ntree=1000,
                           importance=TRUE)
pred<-predict(fit.forest,test)
confusionMatrix(pred,test$genres)
# Initially, .58 accuracy. Remove alternative music and try again.
# Accuracy is .7339 without alternative music. Alternative music is a pretty badly defined genre, and is somewhat of an "other" category, as far as genres go, which would explain some of the misclassifications.
```

### Gradient Boosted Tree
We then created a gradient boosted tree model using XGBoost. A quirk of XGBoost required us to convert genres from factors to numeric variables. Gradient boosted trees are built on the concept of aggregating many shallow (small) decision trees, for high bias and low variance.
```{r xgboost, echo=TRUE, results='hide',message=FALSE, eval=FALSE}
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# xgBoost Trees
library(xgboost)

# Because xgBoost only does numerics, convert from factors to numeric variables
train$key=as.numeric(train$key)
test$key=as.numeric(test$key)
train$mode=as.numeric(train$mode)
test$mode=as.numeric(test$mode)
train$duration_ms=as.numeric(train$duration_ms)
test$duration_ms=as.numeric(test$duration_ms)
train$time_signature=as.numeric(train$time_signature)
test$time_signature=as.numeric(test$time_signature)

# Convert for training dataset
train$rock= ifelse(train$genres=="Rock", 1, 0)
train$indie=ifelse(train$genres=="Indie", 1, 0)
train$country=ifelse(train$genres=="Country", 1, 0)
train$hiphop=ifelse(train$genres=="HipHop", 1, 0)
train$country=ifelse(train$genres=="Country", 1, 0)
train$folk=ifelse(train$genres=="Folk", 1, 0)
train$jazz=ifelse(train$genres=="Jazz", 1, 0)
train$dance=ifelse(train$genres=="Dance", 1, 0)
train$classical=ifelse(train$genres=="Classical", 1, 0)

# Convert for testing dataset
test$rock=ifelse(test$genres=="Rock", 1, 0)
test$indie=ifelse(test$genres=="Indie", 1, 0)
test$country=ifelse(test$genres=="Country", 1, 0)
test$hiphop=ifelse(test$genres=="HipHop", 1, 0)
test$country=ifelse(test$genres=="Country", 1, 0)
test$folk=ifelse(test$genres=="Folk", 1, 0)
test$jazz=ifelse(test$genres=="Jazz", 1, 0)
test$dance=ifelse(test$genres=="Dance", 1, 0)
test$classical=ifelse(test$genres=="Classical", 1, 0)

# Modify train/test datasets
train2 <- train[-c(1,16:23)]
test2 <- test[-c(1,16:23)]

tc <- trainControl(method = "cv",
                   number = 10
                   # ,
                   # summaryFunction = multiClassSummary
)

set.seed(1234)

# Create new model
model2 <- xgboost(data = as.matrix(train2[,-14]),
                  label = as.numeric(train2[,14])-1,
                  tuneLength = 5,
                  eta = 0.2,
                  maxdepth = 15,
                  # nfold = 10,
                  nrounds = 30,
                  objective = "multi:softprob",
                  num_class = 8
                  # ,
                  # num_class = 8
                  # method = "xgbTree",
                  # trControl = tc,
                  # ,
                  # prediction = T
)

model2

predict<-predict(model2,as.matrix(test2[,-14]), reshape=TRUE)
genres <- c("Rock", "Jazz", "Folk", "Indie", "HipHop", "Dance", "Classical", "Country")

class <- apply(predict, 1, which.max)
predicted.class <- sapply(class, FUN = function(x) {
  genres[x]
})

confusionMatrix(predicted.class,test2$genres)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# accuracy is .6855
```
# Results
The Support Vector machine model we created yielded .6935 accuracy, while the Random Forest model initially weighed in at 0.58, and the Gradient Boosted Tree model measured at 0.6855. After further tuning, we were able to improve the Random Forest model's accuracy to .7339. We observed a high variance in sensitivity between genres (ex. Rock, at 0.25, Jazz, at 0.71, and Folk, at 0.86). This is likely owing to the fact that some genres are more general than others, and therefore are more or less easily identified as such. 

# Discussion
By undertaking this task, we experimented with three types of models, and found that the Random Forest model was able to predict genre the best (0.7339). This is an ok solution, but our one main limitation was our inability to pull in a large dataset, due to the restrictions of the Spotify API. Some implications of this task are that music-technology companies can utilize machine learning to quickly classify music into genres using available "already-classified" data. For people conducting similar work, who want to build models to classify music into genres, we would primarily suggest extending the Spotify API, or using a different framework, such that they can access a larger set of data. Our knowledge of the aforementioned machine learning models leads us to believe that if another group tried to conduct this modeling on a larger dataset, their models would be more accurate, up to a certain point.

Another issue is that the question of genre is often a subjective one. A single song can be percieved as occupying multiple genres by a group of people, or even by one person. This is sort of a chicken-and-the-egg problem, in that we could attempt to break down songs into their elements and develop new genres by using machine learning techniques- this would definitely be an interesting problem to tackle in the future. The question of genre, in the abstract sense, is a complex one, hence our imperfect solution. As this was a supervised learning problem, we limited ourselves to the genres given by our Spotify playlists, which are likely hugely imperfect categories. 


# References
1. https://developer.spotify.com/ (Spotify API)
2. QAC Tutors (Specifically Carlo!)
3. Quick-R Website for R syntax and debugging help (https://www.statmethods.net/)




