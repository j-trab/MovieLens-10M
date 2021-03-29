
#CAPSTONE MOVIE RECOMMEDATION

################################################################################

# DATASET PREPARATION

# Create edx set, validation set (final hold-out test set)
##########################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")

if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

if(!require(hrbrthemes)) install.packages("hrbrthemes")

if(!require(ggcorrplot)) install.packages("ggcorrplot")

library(tidyverse)
library(caret)
library(data.table)
library(hrbrthemes)
library(ggcorrplot)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)
##################################
#VARIABLES CLEANING AND EXTRACTION

#Edx head
head(edx)

#Rows and column in the dataset
dim(edx)

#Structure of edx dataset 
str(edx)

#Convert timestamp to review_date
if(!require(lubridate)) install.packages("data.table", repos = "http://cran.us.r-project.org")

review_date <- as_datetime(edx$timestamp) %>% date() %>% year()

#Extract release_date from the Title variable
release_date <- str_extract(edx$title, "\\([0-9]{4}\\)")

release_date <- as.numeric(str_extract(release_date, "[0-9]{4}"))

#Add release_date and review_date variable to the dataset
edx_with_dates <- edx %>% 
  mutate(release_date = release_date, review_date = review_date)

################################################################################
#EXPLORATORY DATA ANALYSIS 

#Average rating 
mean(edx$rating)

#Rating summary statistics 
summary(edx$rating)

#Unique users and movies 
n_distinct(edx$userId)

n_distinct(edx$movieId)

#Most active users 
edx %>% 
  group_by(userId) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

#Most rated movies 
edx %>% 
  group_by(movieId, title) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

#UserId Distribution 
edx %>%
  ggplot(aes(userId)) +
  geom_histogram(fill = "dark red", colour = "dark grey", alpha = 0.9) +
  theme_ipsum()

#MovieId Distribution
edx %>% 
  ggplot(aes(movieId)) +
  geom_histogram(fill = "dark blue", colour = "dark grey", alpha = 0.9) +
  theme_ipsum()

#Rating Distribution 
edx %>%
  ggplot(aes(rating)) +
  geom_bar(alpha = 0.85) +
  geom_vline(xintercept = mean(edx$rating), size = 1, linetype = "dotted",
             colour = "black", alpha = 0.7) +
  labs(title = "Rating distribution") +
  theme_ipsum()

#Rating Distribution by Movie
edx %>% group_by(movieId) %>%
  summarise(ratings = mean(rating)) %>%
  ggplot(aes(ratings)) +
  geom_histogram(colour = "dark blue", alpha = 0.75) +
  geom_vline(xintercept = mean(edx$rating), linetype = "dotted",
             alpha = 0.7) +
  theme_ipsum()

#Rating Distribution by User
edx %>% group_by(userId) %>%
  summarise(ratings = mean(rating)) %>%
  ggplot(aes(ratings)) +
  geom_histogram(colour = "dark red", alpha = 0.75) +
  geom_vline(xintercept = mean(edx$rating), linetype = "dotted",
             alpha = 0.7) +
  theme_ipsum()

################################################################################

#DATA SPlITTING
set.seed(1, sample.kind = "Rounding")

#Create index
tst_index <- createDataPartition(y = edx_with_dates$rating, times = 1, p = 0.2, 
                                 list = FALSE)

#Create train_set / test_set split
train_set <- edx_with_dates %>% slice(-tst_index)
test_set <- edx_with_dates %>% slice(tst_index)

#Make sure all movies and users in the test_set are also in the train_set
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>% 
  semi_join(train_set, by = "userId")

################################################################################

#TIDYVERSE APPROACH

#####################
#Movie + User effects

mu_hat <- mean(train_set$rating)

movie_avgs <- train_set %>%
  group_by(movieId) %>% 
  summarise(bi = mean(rating - mu_hat))

user_avgs <- train_set %>% 
  left_join(movie_avgs, by = "movieId") %>% 
  group_by(userId) %>% 
  summarise(bu = mean(rating - mu_hat - bi))

preds <- test_set %>%
  left_join(movie_avgs, by = "movieId") %>% 
  left_join(user_avgs, by = "userId") %>% 
  mutate(preds = mu_hat + bi + bu) %>% 
  pull(preds)

RMSE(preds, test_set$rating)

#Create results table
rmses <- data.frame(method = "movie + user", Regularized = "NO",
                    CV = "Tr/Tst", RMSE = 0.865932)
rmses

####################################
#Movie + User + Release Date Effects

#Release date distribution
edx %>%
  ggplot(aes(release_date)) +
  geom_histogram(fill = "palegreen4", colour = "dark grey", alpha = 0.9) +
  theme_ipsum()

#Rating distribution by release year 
train_set %>% 
  group_by(release_date) %>% 
  summarise(ratings = mean(rating)) %>% 
  ggplot(aes(ratings)) +
  geom_histogram(bins = 30, color = "palegreen4", alpha = 0.85) +
  geom_vline(xintercept = mu_hat, colour = "black", alpha = 0.7, 
             linetype = "dotted") +
  theme_ipsum()
  
#Release date effect scatterplot
train_set %>% 
  group_by(release_date) %>%
  summarise(ratings = mean(rating)) %>%
  ggplot(aes(release_date, ratings)) +
  geom_point(alpha = 0.7) +
  geom_smooth(colour = "dark green") +
  theme_ipsum()

#Model 
date_avgs <- train_set %>% 
  left_join(movie_avgs, by = "movieId") %>% 
  left_join(user_avgs, by = "userId") %>%
  group_by(release_date) %>% 
  summarise(bd = mean(rating - mu_hat - bi - bu))

#Predictions
preds <- test_set %>%
  left_join(movie_avgs, by = "movieId") %>% 
  left_join(user_avgs, by = "userId") %>% 
  left_join(date_avgs, by = "release_date") %>% 
  mutate(preds = mu_hat + bi + bu + bd) %>% 
  pull(preds)

RMSE(preds, test_set$rating)

rmses <- rmses %>% add_row(method = "movie + user + release_date", 
                           Regularized = "NO", CV = "Tr/Tst",
                           RMSE = 0.8656117)
rmses


###############
#Regularization

#Write a regularization function for the model 
regularization <- function(l, tr, ts) {
  
  #model
  mu_hat <- mean(tr$rating)
  
  bi <- tr %>%
    group_by(movieId) %>%
    summarise(bi = sum(rating - mu_hat) / (n() + l))
  
  bu <- tr %>%
    left_join(bi, by = "movieId") %>%
    group_by(userId) %>%
    summarise(bu = sum(rating - bi - mu_hat) / (n() + l))
  
  bd <- tr %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    group_by(release_date) %>%
    summarise(bd = sum(rating - bi - bu - mu_hat) / (n() + l))
  
  #predictions
  predictions <- ts %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(bd, by = "release_date") %>%
    mutate(preds = mu_hat + bi + bu + bd) %>%
    pull(preds)
  
  return(RMSE(predictions, ts$rating))
}

###################################################
#K-fold cross validation (to pick the best lambda)

#Create cross validation folds
set.seed(1, sample.kind = "Rounding")
folds <- createDataPartition(train_set$rating, times = 5, p = 0.2, list = FALSE)
folds

#Create a 5 fold cross validation function 
best_lambda <- function(l) {
  results <- vector("numeric", 5)
  for (i in 1:5) {
    tr <- train_set[-folds[,i]]
    ts <- train_set[folds[,i]]
    ts <- ts %>%
      semi_join(tr, by = "movieId") %>%
      semi_join(tr, by = "userId")
    
    errors <- regularization(l, tr, ts)
    results[i] <- errors
    
  }
  results
}

#Define a range of possible lambdas 
lambdas <- seq(2, 10, 0.25)

#Apply cross validation to models with varying lambda
res <- sapply(lambdas, best_lambda)

#Results data frame 
pick_lambda <- data.frame(lambdas = lambdas, res = colMeans(res))
pick_lambda

#Plot lambdas against results from cv
pick_lambda %>% ggplot(aes(lambdas, res)) +
  geom_point(alpha = 0.8) +
  theme_ipsum()

#Visualize best lambda
pick_lambda %>% arrange(res)

#Re-run model with the best lambda 
regularization(4.5, train_set, test_set)

#Add entry to results table
rmses <- rmses %>% add_row(method = "movie + user + release_date", 
                           Regularized = "YES", CV = "5-Folds(lambda)",
                           RMSE = 0.8649763)

rmses
##################################################
#Movie + User + Release Date + Review Date Effects

#Review date distribution 
edx %>%
  ggplot(aes(review_date)) +
  geom_histogram(fill = "orchid4", colour = "dark grey", alpha = 0.9) +
  theme_ipsum()

#Rating distribution by review date
train_set %>% 
  group_by(review_date) %>% 
  summarise(ratings = mean(rating)) %>% 
  ggplot(aes(ratings)) +
  geom_histogram(bins = 30, color = "orchid4", alpha = 0.85) +
  geom_vline(xintercept = mu_hat, colour = "black", alpha = 0.7, 
             linetype = "dotted") +
  theme_ipsum()

#Review date effect scatterplot 
train_set %>% 
  group_by(review_date) %>%
  summarise(ratings = mean(rating)) %>%
  ggplot(aes(review_date, ratings)) +
  geom_point(alpha = 0.7) +
  geom_smooth(colour = "orchid4") +
  theme_ipsum()

#Model
review_avgs <- train_set %>% 
  left_join(movie_avgs, by = "movieId") %>% 
  left_join(user_avgs, by = "userId") %>%
  left_join(date_avgs, by = "release_date") %>%
  group_by(review_date) %>% 
  summarise(br = mean(rating - mu_hat - bi - bu - bd))

#Predictions
preds <- test_set %>%
  left_join(movie_avgs, by = "movieId") %>% 
  left_join(user_avgs, by = "userId") %>% 
  left_join(date_avgs, by = "release_date") %>% 
  left_join(review_avgs, by = "review_date")%>%
  mutate(preds = mu_hat + bi + bu + bd + br) %>% 
  pull(preds)

RMSE(preds, test_set$rating)

#Add entry to results table
rmses <- rmses %>% add_row(method = "movie + user + release_date + review_date", 
                           Regularized = "NO", CV = "Tr/Tst",
                           RMSE = 0.8655419)

rmses
###############
#Regularization (user + movie + release_date + review_date)

regularization_rw <- function(l, tr, ts) {
  
  #model
  mu_hat <- mean(tr$rating)
  
  bi <- tr %>%
    group_by(movieId) %>%
    summarise(bi = sum(rating - mu_hat) / (n() + l))
  
  bu <- tr %>%
    left_join(bi, by = "movieId") %>%
    group_by(userId) %>%
    summarise(bu = sum(rating - bi - mu_hat) / (n() + l))
  
  bd <- tr %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    group_by(release_date) %>%
    summarise(bd = sum(rating - bi - bu - mu_hat) / (n() + l))
  
  br <- tr %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(bd, by = "release_date") %>%
    group_by(review_date) %>%
    summarise(br = sum(rating - bi - bu - bd - mu_hat) / (n() + l))
  
  #predictions
  predictions <- ts %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(bd, by = "release_date") %>%
    left_join(br, by = "review_date") %>%
    mutate(preds = mu_hat + bi + bu + bd + br) %>%
    pull(preds)
  
  return(RMSE(predictions, ts$rating))
}

##################################################
#K-fold cross validation (to pick the best lambda)

best_lambda_rw <- function(l) {
  results <- vector("numeric", 5)
  for (i in 1:5) {
    tr <- train_set[-folds[,i]]
    ts <- train_set[folds[,i]]
    ts <- ts %>%
      semi_join(tr, by = "movieId") %>%
      semi_join(tr, by = "userId")
    
    errors <- regularization_rw(l, tr, ts)
    results[i] <- errors
    
  }
  results
}

#5 fold cross validation to pick the best lambda 
lambdas <- seq(4, 6, 0.25)

#Apply cross validation to models with varying lambda
res_rw <- sapply(lambdas, best_lambda_rw)

#Results data frame 
pick_lambda_rw <- data.frame(lambdas = lambdas, res = colMeans(res_rw))
pick_lambda_rw

#Plot results 
pick_lambda_rw %>% ggplot(aes(lambdas, res)) +
  geom_point(alpha = 0.8) +
  theme_ipsum()

#Confirm best lambda
pick_lambda_rw %>% arrange(res)

#Re-run model with the best lambda
regularization_rw(5, train_set, test_set)

#Add result to data table
rmses <- rmses %>% add_row(method = "movie + user + release_date + review_date", 
                           Regularized = "YES", CV = "5-Folds(lambda)",
                           RMSE = 0.8648612)
rmses %>% knitr::kable()

##########################################################
#Movie + User + Release Date + Review Date + Genre Effects

#Explore genres variable 
class(train_set$genres)

n_distinct(train_set$genres)

n_distinct(train_set$movieId)

train_set %>% group_by(genres)

#Rating distribution by genres 
train_set %>% 
  group_by(genres) %>% 
  summarise(ratings = mean(rating)) %>% 
  ggplot(aes(ratings)) +
  geom_histogram(bins = 30, color = "goldenrod3", alpha = 0.85) +
  geom_vline(xintercept = mu_hat, colour = "black", alpha = 0.7, 
             linetype = "dotted") +
  theme_ipsum()

#Regularization Genres
regularization_genres <- function(l, tr, ts) {
  
  #model
  mu_hat <- mean(tr$rating)
  
  bi <- tr %>%
    group_by(movieId) %>%
    summarise(bi = sum(rating - mu_hat) / (n() + l))
  
  bu <- tr %>%
    left_join(bi, by = "movieId") %>%
    group_by(userId) %>%
    summarise(bu = sum(rating - bi - mu_hat) / (n() + l))
  
  bd <- tr %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    group_by(release_date) %>%
    summarise(bd = sum(rating - bi - bu - mu_hat) / (n() + l))
  
  br <- tr %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(bd, by = "release_date") %>%
    group_by(review_date) %>%
    summarise(br = sum(rating - bi - bu - bd - mu_hat) / (n() + l))
  
  bg <- tr %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(bd, by = "release_date") %>%
    left_join(br, by = "review_date") %>%
    group_by(genres) %>%
    summarise(bg = sum(rating - bi - bu - bd - br - mu_hat) / (n() + l))
  
  #predictions
  predictions <- ts %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(bd, by = "release_date") %>%
    left_join(br, by = "review_date") %>%
    left_join(bg, by = "genres") %>%
    mutate(preds = mu_hat + bi + bu + bd + br + bg) %>%
    pull(preds)
  
  return(RMSE(predictions, ts$rating))
}

#########################################
#K-fold cross validation (to pick lambda) 

#5-fold cross validation function
best_lambda_genres <- function(l) {
  results <- vector("numeric", 5)
  for (i in 1:5) {
    tr <- train_set[-folds[,i]]
    ts <- train_set[folds[,i]]
    ts <- ts %>%
      semi_join(tr, by = "movieId") %>%
      semi_join(tr, by = "userId")
    
    errors <- regularization_genres(l, tr, ts)
    results[i] <- errors
    
  }
  results
}

#Define range of possible lambdas
lambdas <- seq(4.5, 6, 0.25)

#Apply range of lambdas
res_genres1 <- sapply(lambdas, best_lambda_genres)

#Results data frame 
pick_lambda_genres <- data.frame(lambdas = lambdas, 
                                 res = colMeans(res_genres1))
#Plot lambdas
pick_lambda_genres %>% ggplot(aes(lambdas, res)) + 
  geom_point(alpha = 0.8)

#Re-run model with the best lambda
regularization_genres(5, train_set, test_set)

#Add result to data table 
rmses <- rmses %>% 
  add_row(method = "movie + user + release_date + review_date + genres",
          Regularized = "YES", CV = "5-Folds(lambda)", RMSE = 0.8645972)

rmses

#################################################################
#Movie + User + Release Date + Review Date + Genre split Effects 

#Split genres 
train_split <- train_set %>% separate_rows(genres, sep = "\\|")

test_split <- test_set %>% separate_rows(genres, sep = "\\|")

#Visualize genre effect with distinct genres 
train_split %>% group_by(genres) %>% 
  summarise(rating = mean(rating)) %>%
  ggplot(aes(genres, rating)) +
  geom_point(alpha = 0.8, colour = "goldenrod3") +
  geom_hline(yintercept = mu_hat, colour = "black", alpha = 0.7, 
             linetype = "dotted") +
  coord_flip() +
  theme_ipsum()

#Create a column for each genre in the dataset 
train_split <- spread(train_split, key = genres, value = timestamp)

test_split <- spread(test_split, key = genres, value = timestamp)

#Binarize genre observations 
#(0 = movie doesn't belong to a given genre, 1 = it does)
train_split[is.na(train_split)] <- 0
train_split[,7:26]<- ifelse(train_split[,7:26] > 0, 1, 0)
train_split <- as.data.frame(train_split) 

test_split[is.na(test_split)] <- 0
test_split[,7:26]<-ifelse(test_split[,7:26] > 0, 1, 0)
test_split <- as.data.frame(test_split)


#Rename no genre column 
train_split <- rename(train_split, "no_genre" = `(no genres listed)`)

test_split <- rename(test_split, "no_genre" = `(no genres listed)`)

#Correlation between individual genres variables
genres_cor<- cor(train_split[,7:26], use = "pairwise.complete")
genres_cor

#Visualize correlation
ggcorrplot(genres_cor, colors = c("goldenrod3", "white", "red")) +
  theme_ipsum(plot_margin = margin(5, 5, 5, 5)) +
  theme(axis.text.x = element_text(angle = 60, hjust = 1, vjust = 1, size = 9),
        axis.text.y = element_text(hjust = 1, vjust = 1, size = 9),
        legend.title = element_text(size = 9),
        legend.text = element_text(size = 7),
        axis.title.x = element_blank(),
        axis.title.y = element_blank())

##########################################
#Try genres effect with one genre (Horror)

#Model / prediction function 
regularization_genres_Horror <- function(l, tr, ts) {
  
  #model
  mu_hat <- mean(tr$rating)
  
  bi <- tr %>%
    group_by(movieId) %>%
    summarise(bi = sum(rating - mu_hat) / (n() + l))
  
  bu <- tr %>%
    left_join(bi, by = "movieId") %>%
    group_by(userId) %>%
    summarise(bu = sum(rating - bi - mu_hat) / (n() + l))
  
  bd <- tr %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    group_by(release_date) %>%
    summarise(bd = sum(rating - bi - bu - mu_hat) / (n() + l))
  
  br <- tr %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(bd, by = "release_date") %>%
    group_by(review_date) %>%
    summarise(br = sum(rating - bi - bu - bd - mu_hat) / (n() + l))
  
  bHorror <- tr %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(bd, by = "release_date") %>%
    left_join(br, by = "review_date") %>%
    group_by(Horror) %>%
    summarise(bHorror = sum(rating - bi - bu - bd - br - mu_hat) / (n() + l))
  
  #predictions
  predictions <- ts %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(bd, by = "release_date") %>%
    left_join(br, by = "review_date") %>%
    left_join(bHorror, by = "Horror") %>%
    mutate(preds = mu_hat + bi + bu + bd + br + bHorror) %>%
    pull(preds)
  
  return(RMSE(predictions, ts$rating))
} 

#Run function with best lambda from the last model without genres
regularization_genres_Horror(5, train_split, test_split)

#Results don't change at all from the last regularized model 
0.8648612

#Number of movies classified as Horror
sum(train_split$Horror)

#Number of movies classified as Action
sum(train_split$Action)

####################################
#Re-run the model with Action effect

#Model and predictions function
regularization_genres_Action <- function(l, tr, ts) {
  
  #model
  mu_hat <- mean(tr$rating)
  
  bi <- tr %>%
    group_by(movieId) %>%
    summarise(bi = sum(rating - mu_hat) / (n() + l))
  
  bu <- tr %>%
    left_join(bi, by = "movieId") %>%
    group_by(userId) %>%
    summarise(bu = sum(rating - bi - mu_hat) / (n() + l))
  
  bd <- tr %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    group_by(release_date) %>%
    summarise(bd = sum(rating - bi - bu - mu_hat) / (n() + l))
  
  br <- tr %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(bd, by = "release_date") %>%
    group_by(review_date) %>%
    summarise(br = sum(rating - bi - bu - bd - mu_hat) / (n() + l))
  
  bAction <- tr %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(bd, by = "release_date") %>%
    left_join(br, by = "review_date") %>%
    group_by(Action) %>%
    summarise(bAction = sum(rating - bi - bu - bd - br - mu_hat) / (n() + l))
  
  #predictions
  predictions <- ts %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(bd, by = "release_date") %>%
    left_join(br, by = "review_date") %>%
    left_join(bAction, by = "Action") %>%
    mutate(preds = mu_hat + bi + bu + bd + br + bAction) %>%
    pull(preds)
  
  return(RMSE(predictions, ts$rating))
}

#Run function
regularization_genres_Action(5, train_split, test_split)

#Results improve with the Horror genre
0.8648347

###################################################################
#Build a model only with genres with relevant number of observation

#Build a genres data frame 
genres <- train_split[,8:25]

#Number of observations per genre
genres_sum <- sapply(genres, function(n){
  res <- sum(n)
}) 

#Order genres by number of observations and filter out non-significant genres
data.frame(genres_sum) %>% 
  filter(. > 999999) %>% 
  arrange(desc(.))

####################################
#Build model with significant genres

#Regularized model and predictions
regularization_genres_split <- function(l, tr, ts) {
  
  #model
  mu_hat <- mean(tr$rating)
  
  bi <- tr %>%
    group_by(movieId) %>%
    summarise(bi = sum(rating - mu_hat) / (n() + l))
  
  bu <- tr %>%
    left_join(bi, by = "movieId") %>%
    group_by(userId) %>%
    summarise(bu = sum(rating - bi - mu_hat) / (n() + l))
  
  bd <- tr %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    group_by(release_date) %>%
    summarise(bd = sum(rating - bi - bu - mu_hat) / (n() + l))
  
  br <- tr %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(bd, by = "release_date") %>%
    group_by(review_date) %>%
    summarise(br = sum(rating - bi - bu - bd - mu_hat) / (n() + l))
  
  bDrama <- tr %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(bd, by = "release_date") %>%
    left_join(br, by = "review_date") %>%
    group_by(Drama) %>%
    summarise(bDrama = sum(rating - bi - bu - bd - br - mu_hat) / (n() + l))
  
  bComedy <- tr %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(bd, by = "release_date") %>%
    left_join(br, by = "review_date") %>%
    left_join(bDrama, by = "Drama") %>%
    group_by(Comedy) %>%
    summarise(bComedy = sum(rating - bi - bu - bd - br - bDrama - mu_hat) / 
                (n() + l))
  
  bAction <- tr %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(bd, by = "release_date") %>%
    left_join(br, by = "review_date") %>%
    left_join(bDrama, by = "Drama") %>%
    left_join(bComedy, by = "Comedy") %>%
    group_by(Action) %>%
    summarise(bAction = sum(rating - bi - bu - bd - br - bDrama - bComedy -
                              mu_hat) / (n() + l))
  
  bThriller <- tr %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(bd, by = "release_date") %>%
    left_join(br, by = "review_date") %>%
    left_join(bDrama, by = "Drama") %>%
    left_join(bComedy, by = "Comedy") %>%
    left_join(bAction, by = "Action") %>% 
    group_by(Thriller) %>%
    summarise(bThriller = sum(rating - bi - bu - bd - br - bDrama - bComedy -
                                bAction - mu_hat) / (n() + l))
  
  bAdventure <- tr %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(bd, by = "release_date") %>%
    left_join(br, by = "review_date") %>%
    left_join(bDrama, by = "Drama") %>%
    left_join(bComedy, by = "Comedy") %>%
    left_join(bAction, by = "Action") %>% 
    left_join(bThriller, by = "Thriller") %>%
    group_by(Adventure) %>%
    summarise(bAdventure = sum(rating - bi - bu - bd - br - bDrama - bComedy -
                                 bAction - bThriller - mu_hat) / (n() + l))
  
  bRomance <- tr %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(bd, by = "release_date") %>%
    left_join(br, by = "review_date") %>%
    left_join(bDrama, by = "Drama") %>%
    left_join(bComedy, by = "Comedy") %>%
    left_join(bAction, by = "Action") %>% 
    left_join(bThriller, by = "Thriller") %>%
    left_join(bAdventure, by = "Adventure") %>%
    group_by(Romance) %>%
    summarise(bRomance = sum(rating - bi - bu - bd - br - bDrama - bComedy -
                               bThriller - bAction - bAdventure -
                               mu_hat) / (n() + l))
  
  bSci <- tr %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(bd, by = "release_date") %>%
    left_join(br, by = "review_date") %>%
    left_join(bDrama, by = "Drama") %>%
    left_join(bComedy, by = "Comedy") %>%
    left_join(bAction, by = "Action") %>% 
    left_join(bThriller, by = "Thriller") %>%
    left_join(bAdventure, by = "Adventure") %>%
    left_join(bRomance, by = "Romance") %>%
    group_by(`Sci-Fi`) %>%
    summarise(bSci = sum(rating - bi - bu - bd - br - bDrama - bComedy -
                           bThriller - bAction - bAdventure - bRomance -
                           mu_hat) / (n() + l))
  
  bCrime <- tr %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(bd, by = "release_date") %>%
    left_join(br, by = "review_date") %>%
    left_join(bDrama, by = "Drama") %>%
    left_join(bComedy, by = "Comedy") %>%
    left_join(bAction, by = "Action") %>% 
    left_join(bThriller, by = "Thriller") %>%
    left_join(bAdventure, by = "Adventure") %>%
    left_join(bRomance, by = "Romance") %>%
    left_join(bSci, by = "Sci-Fi") %>%
    group_by(Crime) %>%
    summarise(bCrime = sum(rating - bi - bu - bd - br - bDrama - bComedy -
                             bThriller - bAction - bAdventure - bRomance - bSci -
                             mu_hat) / (n() + l))
  
  #predictions
  predictions <- ts %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(bd, by = "release_date") %>%
    left_join(br, by = "review_date") %>%
    left_join(bDrama, by = "Drama") %>%
    left_join(bComedy, by = "Comedy") %>%
    left_join(bAction, by = "Action") %>%
    left_join(bThriller, by = "Thriller") %>%
    left_join(bAdventure, by = "Adventure") %>% 
    left_join(bRomance, by = "Romance") %>%
    left_join(bSci, by = "Sci-Fi") %>%
    left_join(bCrime, by = "Crime") %>%
    mutate(preds = mu_hat + bi + bu + bd + br + bDrama + bComedy + bAction +
             bThriller + bAdventure + bRomance + bSci + bCrime) %>%
    pull(preds)
  
  return(RMSE(predictions, ts$rating))
}

#Run model with lambda = 5
regularization_genres_split(5, train_split, test_split)

##############################################
#K-fold cross validation (to pick best lambda)

#5-fold cross validation function
best_lambda_genres_split <- function(l) {
  results <- vector("numeric", 5)
  for (i in 1:5) {
    tr <- train_split %>% slice(-folds[,i])
    ts <- train_split %>% slice(folds[,i])
    ts <- ts %>%
      semi_join(tr, by = "movieId") %>%
      semi_join(tr, by = "userId")
    
    errors <- regularization_genres_split(l, tr, ts)
    results[i] <- errors
    
  }
  results
}

#Range of lambdas
lambdas <- seq(4.5, 5.75, 0.25)

#Run cross validation
res_genres_split <- sapply(lambdas, best_lambda_genres_split)

#Results data frame
pick_lambda_genres_split <- data.frame(lambdas = lambdas, 
                                       rmses = colMeans(res_genres_split))

#Plot rmses against lambda
pick_lambda_genres_split %>% ggplot(aes(lambdas, rmses)) +
  geom_point(alpha = 0.8) +
  theme_ipsum()

#Single out best lambda
pick_lambda_genres_split %>% arrange(rmses)

#Run model with best lambda
regularization_genres_split(5, train_split, test_split)

#Add entry to results table
rmses <- rmses %>%
  add_row(method = "6 Movie + user + release date + review date + g(split)",
          Regularized = "YES", CV = "5-Folds(lambda)", RMSE = 0.8647848)

rmses %>% knitr::kable()

################################################################################
#MATRIX FACTORIZATION

#################
#Download package
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")

#Load package
library(recosystem)

#set seed for randomized process
set.seed(1, sample.kind = "Rounding")

##############################
#Format data for package usage 

#Specify index1 = TRUE since our train_set userId first entry is 1
train_reco <- data_memory(train_set$userId, train_set$movieId, train_set$rating,
                          index1 = TRUE)

test_reco <- data_memory(test_set$userId, test_set$movieId, index1 = TRUE)

#Call to Reco function
r <- Reco()

#Tune hyper-parameters 
opts <- r$tune(train_reco, opts = list(dim = c(10, 20, 30), lrate = c(0.1, 0.2),
                                       nthread = 2, niter = 5))

#train algorithm 
r$train(train_data = train_reco, opts = c(opts$min, nthread = 2, niter = 100))

#predict 
predictions_reco <- r$predict(test_reco, out_pred = out_memory()) 

#Rmses
RMSE(predictions_reco, test_set$rating)

#Result Rmses
0.8019637

#Re-tune with different hyper-parameters
opts1 <- r$tune(train_reco, opts = list(dim = 30, lrate = c(0.1, 0.2), 
                                        costp_l1 = 0, costp_l2 = 0.01,
                                        costq_l1 = 0, costq_l2 = c(0.1, 0.2),
                                        nthread = 2, niter = 20))

#Train with new best hyper-parameters
r$train(train_reco, opts = c(opts1$min, nthread = 2, niter = 100))

#New predictions 
predictions_reco1 <- r$predict(test_reco, out_pred = out_memory())

#New RMSES
RMSE(predictions_reco1, test_set$rating)

#Add best (2nd) rmses to results table 
rmses <- rmses %>% add_row(method = "Matrix Factorization (Recosystem)", 
                           Regularized = "YES", CV = "5-Folds (Tune)",
                           RMSE = 0.7936243)

#Results table in kniter format
rmses %>% knitr::kable()

################################################################################
#VALIDATION 

#Carrie out validation with our last recosystem model

#Set seed
set.seed(1, sample.kind = "Rounding")

#Format train set
edx_reco <- data_memory(edx$userId, edx$movieId, edx$rating,
                        index1 = TRUE)
#Format test set
validation_reco <- data_memory(validation$userId, validation$movieId,
                               index1 = TRUE)

#Call to reco function 
r_validation <- Reco()

#Train model
r_validation$train(train_data = edx_reco, 
                   opts = c(opts1$min, nthread = 2, niter = 100))

#Predictions on the validation set
predictions_validation <- r_validation$predict(validation_reco, 
                                               out_pred = out_memory())

#Final matrix factorization on the validation set
RMSE(predictions_validation, validation$rating)

#Add validated RMSE to results table 
rmses <- rmses %>% add_row(method = " 8 Matrix Factorization (Validated)", 
                           Regularized = "YES", CV = "5-Folds (Tuned)",
                           RMSE = 0.7825019)

#Results table in knitr format
rmses %>% knitr::kable()



