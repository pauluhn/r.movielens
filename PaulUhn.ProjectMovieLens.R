##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings |>
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp),
         date = as_datetime(timestamp)) |> # calculating date from timestamp
  select(userId, movieId, rating, date)    # removing timestamp

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies |>
  separate_wider_regex(title, c(title = ".*", " \\(", year = "\\d{4}", "\\)")) |> # splitting title and year
  mutate(movieId = as.integer(movieId),
         year = as.integer(year))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp |>
  semi_join(edx, by = "movieId") |>
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test, by = join_by(userId, movieId, rating, date, title, genres))
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed, movies_file, ratings_file)

edx |> as_tibble()

edx |>
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

# generate table of users and movie ratings
keep <- edx |> 
  dplyr::count(movieId) |> 
  slice_max(n, n = 4) |> 
  pull(movieId)
tab <- edx |> 
  filter(movieId %in% keep) |> 
  filter(userId %in% c(18:34)) |> # find an "interesting" set of users
  select(userId, title, rating) |> 
  mutate(title = str_remove(title, ", The")) |>
  pivot_wider(names_from = "title", values_from = "rating")
if(!require(kableExtra)) install.packages("kableExtra")
library(kableExtra)
if(knitr::is_html_output()) {
  knitr::kable(tab, "html") |>
    kableExtra::kable_styling(bootstrap_options = "striped", full_width = FALSE)
} else {
  knitr::kable(tab, "latex", booktabs = TRUE) |>
    kableExtra::kable_styling(font_size = 8)
}
rm(keep, tab)

# generate a dotplot of users/movies
set.seed(2023)
users <- sample(unique(edx$userId), 100)
if(!require(rafalib)) install.packages("rafalib")
library(rafalib)
rafalib::mypar()
edx |> 
  filter(userId %in% users) |> 
  select(userId, movieId, rating) |>
  mutate(rating = 1) |>
  pivot_wider(names_from = movieId, values_from = rating) |> 
  (\(mat) mat[, sample(ncol(mat), 100)])() |>
  as.matrix() |> 
  t() |>
  image(1:100, 1:100, z = _, xlab = "Movies", ylab = "Users")
rm(users)

# generate a barplot of movies and users
p1 <- edx |> 
  count(movieId) |> 
  ggplot(aes(n)) + 
  geom_histogram(bins = 25, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")
p2 <- edx |> 
  count(userId) |> 
  ggplot(aes(n)) + 
  geom_histogram(bins = 25, color = "black") + 
  scale_x_log10() + 
  ggtitle("Users")
if(!require(gridExtra)) install.packages("gridExtra")
library(gridExtra)
gridExtra::grid.arrange(p1, p2, ncol = 2)
rm(p1, p2)

# split the data into training and test sets
set.seed(2023)
indexes <- split(1:nrow(edx), edx$userId)
# 20% of ratings into test set
test_ind <- sapply(indexes, function(ind) sample(ind, ceiling(length(ind)*.2))) |>
  unlist(use.names = TRUE) |> sort()
test_set <- edx[test_ind,]   # 1827788
train_set <- edx[-test_ind,] # 7172267
# remove entries
test_set <- test_set |> 
  semi_join(train_set, by = "movieId") # 1827746
train_set <- train_set |> 
  semi_join(test_set, by = "movieId")  # 7170310
# matrix
y <- train_set |>
  select(movieId, userId, rating) |>
  pivot_wider(names_from = movieId, values_from = rating) 
rnames <- y$userId
y <- as.matrix(y[,-1]) # row=user, col=movie
rownames(y) <- rnames
# movie id to title
movie_map <- train_set |> 
  select(movieId, title) |> 
  distinct(movieId, .keep_all = TRUE)
rm(indexes, test_ind, rnames)

# loss function
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings-predicted_ratings)^2))
}

# first model
mu <- mean(y, na.rm = TRUE) # mu_hat
trivial_rmse <- RMSE(test_set$rating, mu)
rmse_results <- tibble(method = "Just the average", RMSE = trivial_rmse)
rm(trivial_rmse)

# parameter b_i, bias movie (item)
# fit <- lm(rating ~ as.factor(movieId), data = edx) # very slow
b_i <- colMeans(y - mu, na.rm = TRUE) # b_i_hat
hist(b_i)
fit_movies <- data.frame(movieId = as.integer(colnames(y)), 
                         mu = mu, 
                         b_i = b_i)
movie_effect_rmse <- left_join(test_set, fit_movies, by = "movieId") |> 
  mutate(pred = mu + b_i) |> 
  summarize(rmse = RMSE(rating, pred)) |>
  pull(rmse)
rmse_results <- rmse_results |>
  add_row(method = "Movie effect", RMSE = movie_effect_rmse)
rm(movie_effect_rmse)

# parameter b_u, bias user
# fit <- lm(rating ~ as.factor(movieId) + as.factor(userId), data = edx) # very slow
b_u <- rowMeans(sweep(y - mu, 2, b_i), na.rm = TRUE) # b_u_hat
hist(b_u)
fit_users <- data.frame(userId = as.integer(rownames(y)), 
                        b_u = b_u)
user_effect_rmse <- left_join(test_set, fit_movies, by = "movieId") |> 
  left_join(fit_users, by = "userId") |>
  mutate(pred = mu + b_i + b_u) |> 
  summarize(rmse = RMSE(rating, pred)) |>
  pull(rmse)
rmse_results <- rmse_results |>
  add_row(method = "Movie + User effect", RMSE = user_effect_rmse)
rm(user_effect_rmse, b_i, b_u)

# explore regularization
n <- colSums(!is.na(y))
fit_movies$n <- n
best <- fit_movies |> left_join(movie_map, by = "movieId") |> 
  mutate(average_rating = mu + b_i) |>
  filter(average_rating > 4.5 & n > 1) 
test_set |> 
  group_by(movieId) |>
  summarize(test_set_averge_rating = mean(rating)) |>
  right_join(best, by = "movieId") |>
  select(title, average_rating, n, test_set_averge_rating) 
rm(best)

# calculate the rmses with different lambdas
lambdas <- seq(0, 10, 0.1)
sums <- colSums(y - mu, na.rm = TRUE)
rmses <- sapply(lambdas, function(lambda) {
  b_i <- sums / (n + lambda)
  fit_movies$b_i <- b_i
  left_join(test_set, fit_movies, by = "movieId") |> 
    mutate(pred = mu + b_i) |> 
    summarize(rmse = RMSE(rating, pred)) |>
    pull(rmse)
})
# plot and find the lambda that minimizes rmse
plot(lambdas, rmses, type = "l")
lambda <- lambdas[which.min(rmses)]
# calculate regularized b_i
fit_movies$b_i_reg <- sums / (n + lambda)
rm(n, sums, lambda)

# show movies using regularized b_i
best <- fit_movies |> left_join(movie_map, by = "movieId") |> 
  top_n(5, b_i_reg) |> 
  arrange(desc(b_i_reg)) |>
  mutate(average_rating = mu + b_i_reg)
test_set |> 
  group_by(movieId) |>
  summarize(test_set_averge_rating = mean(rating)) |>
  right_join(best, by = "movieId") |>
  select(title, average_rating, n, test_set_averge_rating) 
rm(best)

# compute regularized movie effect
reg_movie_rmse <- left_join(test_set, fit_movies, by = "movieId") |> 
  mutate(pred = mu + b_i_reg) |> 
  summarize(rmse = RMSE(rating, pred)) |>
  pull(rmse)
rmse_results <- rmse_results |>
  add_row(method = "Regularized Movie effect", RMSE = reg_movie_rmse)
rm(reg_movie_rmse)
# compuete regularized movie effect with user effect
reg_movie_user_rmse <- left_join(test_set, fit_movies, by = "movieId") |> 
  left_join(fit_users, by = "userId") |>
  mutate(pred = mu + b_i_reg + b_u) |> 
  summarize(rmse = RMSE(rating, pred)) |>
  pull(rmse)
rmse_results <- rmse_results |>
  add_row(method = "Regularized Movie + User effect", RMSE = reg_movie_user_rmse)
rm(reg_movie_user_rmse)

# calculate lambda/rmses for regularizing user effect
m <- rowSums(!is.na(y))
fit_users$m <- m
lambdas <- seq(0, 10, 0.1)
sums <- rowSums(sweep(y - mu, 2, fit_movies$b_i_reg), na.rm = TRUE)
rmses <- sapply(lambdas, function(lambda) {
  b_u <- sums / (m + lambda)
  fit_users$b_u <- b_u
  left_join(test_set, fit_movies, by = "movieId") |> 
    left_join(fit_users, by = "userId") |> 
    mutate(pred = mu + b_i_reg + b_u) |> 
    summarize(rmse = RMSE(rating, pred)) |>
    pull(rmse)
})
plot(lambdas, rmses, type = "l")
lambda <- lambdas[which.min(rmses)]
# compute rmse for regularized movie effect and regularized user effect
fit_users$b_u_reg <- sums / (m + lambda)
reg_user_rmse <- left_join(test_set, fit_movies, by = "movieId") |> 
  left_join(fit_users, by = "userId") |>
  mutate(pred = mu + b_i_reg + b_u_reg) |> 
  summarize(rmse = RMSE(rating, pred)) |>
  pull(rmse)
rmse_results <- rmse_results |>
  add_row(method = "Reg Movie + Reg User effect", RMSE = reg_user_rmse)
rm(m, sums, reg_user_rmse, lambdas, rmses, lambda)

# explore rating per year (rate) vs user rating using edx
min(edx$year) # 1915
max(edx$year) # 2008
edx |>
  group_by(movieId) |>
  summarize(n = n(), 
            years = 2023 - first(year),
            title = title[1],
            rating = mean(rating)) |>
  mutate(rate = n / years) |>
  ggplot(aes(rate, rating)) +
  geom_point() +
  geom_smooth()

# calculate rate on training set
rate_to_rating <- train_set |>
  mutate(rating = rating - mu) |>
  group_by(movieId) |>
  summarize(n = n(), 
            years = 2023 - first(year),
            title = title[1],
            rating = mean(rating)) |>
  mutate(rate = n / years)
# confirm trend
rate_to_rating |>
  ggplot(aes(rate, rating)) +
  geom_point() +
  geom_smooth()

# saving for future
rate_map <- rate_to_rating |> select(movieId, rate)

# prep test set for rate rmse
test_set_rate <- test_set |>
  left_join(rate_map, by = "movieId")
# compare training methods
fit_rate_glm <- train(rating ~ rate, method = "glm", data = rate_to_rating)
b_r_glm <- predict(fit_rate_glm, test_set_rate)
data.frame(pred = mu + b_r_glm, rating = test_set$rating) |>
  summarize(rmse = RMSE(rating, pred)) |>
  pull(rmse) # 1.061823
fit_rate_gam <- train(rating ~ rate, method = "gamLoess", data = rate_to_rating)
b_r_gam <- predict(fit_rate_gam, test_set_rate)
data.frame(pred = mu + b_r_gam, rating = test_set$rating) |>
  summarize(rmse = RMSE(rating, pred)) |>
  pull(rmse) # 1.048869
fit_rate_knn <- train(rating ~ rate, method = "knn", data = rate_to_rating)
b_r_knn <- predict(fit_rate_knn, test_set_rate)
data.frame(pred = mu + b_r_knn, rating = test_set$rating) |>
  summarize(rmse = RMSE(rating, pred)) |>
  pull(rmse) # 1.037032

# calculate rate on training set (use reg movie reg user)
rate_to_rating <- train_set |>
  left_join(fit_movies, by = "movieId") |> 
  left_join(fit_users, by = "userId") |>
  mutate(rating = rating - mu - b_i_reg - b_u_reg) |>
  group_by(movieId) |>
  summarize(n = n(), 
            years = 2023 - first(year),
            title = title[1],
            rating = mean(rating)) |>
  mutate(rate = n / years)
# prep test set
test_set_rate <- test_set |>
  left_join(fit_movies, by = "movieId") |> 
  left_join(fit_users, by = "userId") |>
  left_join(rate_map, by = "movieId")
# compare training methods
fit_rate_glm <- train(rating ~ rate, method = "glm", data = rate_to_rating)
fit_rate_gam <- train(rating ~ rate, method = "gamLoess", data = rate_to_rating)
fit_rate_knn <- train(rating ~ rate, method = "knn", data = rate_to_rating)
b_r_glm <- predict(fit_rate_glm, test_set_rate)
b_r_gam <- predict(fit_rate_gam, test_set_rate)
b_r_knn <- predict(fit_rate_knn, test_set_rate)
test_set_b_r <- test_set_rate |> 
  mutate(pred_glm = mu + b_i_reg + b_u_reg + b_r_glm,
         pred_gam = mu + b_i_reg + b_u_reg + b_r_gam,
         pred_knn = mu + b_i_reg + b_u_reg + b_r_knn) |>
  summarize(rmse_glm = RMSE(rating, pred_glm), # 0.8664498
            rmse_gam = RMSE(rating, pred_gam), # 0.8664498
            rmse_knn = RMSE(rating, pred_knn)) # 0.8660144
min(test_set_b_r) # 0.8660144

# saving for future
fit_rate <- fit_rate_knn

rm(rate_to_rating, test_set_rate, fit_rate_glm, fit_rate_gam, fit_rate_knn, b_r_glm, b_r_gam, b_r_knn, test_set_b_r)

# explore date vs user rating using edx
edx |>
  mutate(date = round_date(date, unit = "week")) |>
  group_by(date) |>
  summarize(rating = mean(rating)) |>
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth()

# prep the training set
date_to_rating <- train_set |>
  mutate(rating = rating - mu,
         date = round_date(date, unit = "week")) |>
  group_by(date) |>
  summarize(rating = mean(rating))
# confirm trend
date_to_rating |>
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth()
# prep the test set
test_set_date <- test_set |>
  mutate(date = round_date(date, unit = "week"))

# compare training methods
fit_date_glm <- train(rating ~ date, method = "glm", data = date_to_rating)
fit_date_gam <- train(rating ~ date, method = "gamLoess", data = date_to_rating)
fit_date_knn <- train(rating ~ date, method = "knn", data = date_to_rating)

b_d_glm <- predict(fit_date_glm, test_set_date)
b_d_gam <- predict(fit_date_gam, test_set_date)
b_d_knn <- predict(fit_date_knn, test_set_date)

data.frame(rating = test_set_date$rating,
           pred_glm = mu + b_d_glm, 
           pred_gam = mu + b_d_gam,
           pred_knn = mu + b_d_knn) |>
  summarize(rmse_glm = RMSE(rating, pred_glm), # 1.060066
            rmse_gam = RMSE(rating, pred_gam), # 1.059555
            rmse_knn = RMSE(rating, pred_knn)) # 1.058381

# calculate date on training set (use movie user rate)
date_to_rating <- train_set |>
  left_join(fit_movies, by = "movieId") |> 
  left_join(fit_users, by = "userId") |>
  left_join(rate_map, by = "movieId")
date_to_rating <- date_to_rating |>
  mutate(b_r = predict(fit_rate, newdata = date_to_rating),
         rating = rating - mu - b_i_reg - b_u_reg - b_r,
         date = round_date(date, unit = "week")) |>
  group_by(date) |>
  summarize(rating = mean(rating))

# prep test set
test_set_date <- test_set |>
  left_join(fit_movies, by = "movieId") |> 
  left_join(fit_users, by = "userId") |>
  left_join(rate_map, by = "movieId")
test_set_date <- test_set_date |>
  mutate(b_r = predict(fit_rate, newdata = test_set_date),
         date = round_date(date, unit = "week"))

# compare training methods
fit_date_glm <- train(rating ~ date, method = "glm", data = date_to_rating)
fit_date_gam <- train(rating ~ date, method = "gamLoess", data = date_to_rating)
fit_date_knn <- train(rating ~ date, method = "knn", data = date_to_rating)

b_d_glm <- predict(fit_date_glm, test_set_date)
b_d_gam <- predict(fit_date_gam, test_set_date)
b_d_knn <- predict(fit_date_knn, test_set_date)

test_set_b_d <- test_set_date |> 
  mutate(pred_glm = mu + b_i_reg + b_u_reg + b_r + b_d_glm,
         pred_gam = mu + b_i_reg + b_u_reg + b_r + b_d_gam,
         pred_knn = mu + b_i_reg + b_u_reg + b_r + b_d_knn) |>
  summarize(rmse_glm = RMSE(rating, pred_glm),
            rmse_gam = RMSE(rating, pred_gam),
            rmse_knn = RMSE(rating, pred_knn))
test_set_b_d
min(test_set_b_d) # knn was best

# saving for future
fit_date <- fit_date_knn

rm(date_to_rating, test_set_date, fit_date_glm, fit_date_gam, fit_date_knn, b_d_glm, b_d_gam, b_d_knn, test_set_b_d)

# explore genre vs ratings using edx
edx |>
  group_by(genres) |>
  summarize(n = n(), 
            avg = mean(rating), 
            se = sd(rating)/sqrt(n())) |>
  filter(n >= 1000) |>
  mutate(genres = reorder(genres, avg)) |>
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 4))

# prep training
genre_map <- train_set |>
  mutate(rating = rating - mu) |>
  group_by(genres) |>
  summarize(n = n(), 
            b_g = mean(rating)) |>
  filter(n >= 1000) |> # 783 -> 420 genres
  select(genres, b_g)

# prep test
test_set |>
  semi_join(genre_map, by = "genres") |> # keep matching genres
  left_join(genre_map, by = "genres") |> # add b_g
  mutate(pred = mu + b_g) |>
  summarize(rmse = RMSE(rating, pred)) |>
  pull(rmse)

# prep training #2
genre_to_rating <- train_set |>
  left_join(fit_movies, by = "movieId") |> 
  left_join(fit_users, by = "userId") |>
  left_join(rate_map, by = "movieId") |>
  mutate(date = round_date(date, unit = "week"))
genre_map <- genre_to_rating |>
  mutate(b_r = predict(fit_rate, newdata = genre_to_rating),
         b_d = predict(fit_date, newdata = genre_to_rating),
         rating = rating - mu - b_i_reg - b_u_reg - b_r - b_d) |>
  group_by(genres) |>
  summarize(n = n(), 
            b_g = mean(rating)) |>
  filter(n >= 1000) |> # 783 -> 420 genres
  select(genres, b_g)
rm(genre_to_rating)

# prep test #2
test_set_genre <- test_set |>
  left_join(fit_movies, by = "movieId") |> 
  left_join(fit_users, by = "userId") |>
  left_join(rate_map, by = "movieId") |>
  mutate(date = round_date(date, unit = "week")) |>
  semi_join(genre_map, by = "genres") |> # keep matching genres
  left_join(genre_map, by = "genres")    # add b_g
test_set_genre |>
    mutate(b_r = predict(fit_rate, newdata = test_set_genre),
         b_d = predict(fit_date, newdata = test_set_genre),
         pred = mu + b_i_reg + b_u_reg + b_r + b_d + b_g) |>
  summarize(rmse = RMSE(rating, pred)) |>
  pull(rmse)
rm(test_set_genre)

# explore (split) genre vs ratings using edx
genre_count <- edx |>
  group_by(movieId) |>
  summarize(genres = first(genres)) |>
  mutate(genre_count = str_count(genres, "\\|") + 1)
range(genre_count$genre_count)
genre_count |>
  group_by(genre_count) |>
  summarize(n = n())
rm(genre_count)
  
edx |>
  separate_longer_delim(genres, "|") |>
  group_by(genres) |>
  summarize(n = n(), 
            avg = mean(rating), 
            se = sd(rating)/sqrt(n())) |>
  filter(n >= 1000) |>
  mutate(genres = reorder(genres, avg)) |>
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# prep training #3
genre_map <- train_set |>
  mutate(rating = rating - mu) |>
  separate_longer_delim(genres, "|") |>
  group_by(genres) |>
  summarize(n = n(), 
            b_g = mean(rating)) |>
  select(genres, b_g)

# prep test #3
test_set |>
  separate_longer_delim(genres, "|") |>
  left_join(genre_map, by = "genres") |> # add b_g
  mutate(pred = mu + b_g) |>
  group_by(movieId) |>
  summarize(pred = mean(pred),
            rating = first(rating)) |>
  ungroup() |>
  summarize(rmse = RMSE(rating, pred)) |>
  pull(rmse)

# prep training #4
genre_to_rating <- train_set |>
  left_join(fit_movies, by = "movieId") |> 
  left_join(fit_users, by = "userId") |>
  left_join(rate_map, by = "movieId") |>
  mutate(date = round_date(date, unit = "week"))
genre_map <- genre_to_rating |>
  mutate(b_r = predict(fit_rate, newdata = genre_to_rating),
         b_d = predict(fit_date, newdata = genre_to_rating),
         rating = rating - mu - b_i_reg - b_u_reg - b_r - b_d,
         genre_count = str_count(genres, "\\|") + 1, ###
         rating = rating / genre_count) |> ###
  separate_longer_delim(genres, "|") |>
  group_by(genres) |>
  summarize(n = n(), 
            b_g = mean(rating)) |>
  select(genres, b_g)
rm(genre_to_rating)

# prep test #4
test_set_genre <- test_set |>
  left_join(fit_movies, by = "movieId") |> 
  left_join(fit_users, by = "userId") |>
  left_join(rate_map, by = "movieId") |>
  mutate(date = round_date(date, unit = "week"))
test_set_genre |>
  mutate(b_r = predict(fit_rate, newdata = test_set_genre),
         b_d = predict(fit_date, newdata = test_set_genre)) |>
  separate_longer_delim(genres, "|") |>
  left_join(genre_map, by = "genres") |> # add b_g
  #mutate(pred = mu + b_i_reg + b_u_reg + b_r + b_d + b_g) |>
  mutate(pred = mu + b_i_reg + b_u_reg + b_r + b_d) |>
  group_by(movieId) |>
  summarize(pred = mean(pred),
            rating = first(rating),
            rating = rating + sum(b_g)) |> ###
  ungroup() |>
  summarize(rmse = RMSE(rating, pred)) |>
  pull(rmse)
rm(test_set_genre)

# final edx training

# mu
mu <- mean(edx$rating, na.rm = TRUE)
mu
# b_i_reg - lambda = 2.3
fit_movies <- edx |>
  mutate(rating = rating - mu) |>
  group_by(movieId) |>
  summarize(b_i_reg = sum(rating) / (n() + 2.3))
# b_u_reg - lambda = 4.8
fit_users <- edx |>
  left_join(fit_movies, by = "movieId") |>
  mutate(rating = rating - mu - b_i_reg) |>
  group_by(userId) |>
  summarize(b_u_reg = sum(rating) / (n() + 4.8))
# b_r - knn
rate_to_rating <- edx |>
  left_join(fit_movies, by = "movieId") |>
  left_join(fit_users, by = "userId") |>
  mutate(rating = rating - mu - b_i_reg - b_u_reg) |>
  group_by(movieId) |>
  summarize(rating = mean(rating),
            rate = n() / (2023 - first(year)))
rate_map <- rate_to_rating |> 
  select(movieId, rate)
fit_rate <- train(rating ~ rate, method = "knn", data = rate_to_rating)
# b_d - knn
date_to_rating <- edx |>
  left_join(fit_movies, by = "movieId") |>
  left_join(fit_users, by = "userId") |>
  left_join(rate_map, by = "movieId")
date_to_rating <- date_to_rating |>
  mutate(b_r = predict(fit_rate, date_to_rating),
         rating = rating - mu - b_i_reg - b_u_reg - b_r,
         date = round_date(date, unit = "week")) |>
  group_by(date) |>
  summarize(rating = mean(rating))
fit_date <- train(rating ~ date, method = "knn", data = date_to_rating)
# b_g
genre_to_rating <- edx |>
  left_join(fit_movies, by = "movieId") |> 
  left_join(fit_users, by = "userId") |>
  left_join(rate_map, by = "movieId") |>
  mutate(date = round_date(date, unit = "week"))
genre_map <- genre_to_rating |>
  mutate(b_r = predict(fit_rate, genre_to_rating),
         b_d = predict(fit_date, genre_to_rating),
         rating = rating - mu - b_i_reg - b_u_reg - b_r - b_d) |>
  group_by(genres) |>
  summarize(n = n(), 
            b_g = mean(rating)) |>
  filter(n >= 1000) |>
  select(genres, b_g)

# final holdout test

final_holdout_test_set <- final_holdout_test |>
  left_join(fit_movies, by = "movieId") |> 
  left_join(fit_users, by = "userId") |>
  left_join(rate_map, by = "movieId") |>
  mutate(date = round_date(date, unit = "week")) |>
  semi_join(genre_map, by = "genres") |>
  left_join(genre_map, by = "genres")
rmse_final_holdout_test <- final_holdout_test_set |>
  mutate(b_r = predict(fit_rate, final_holdout_test_set),
         b_d = predict(fit_date, final_holdout_test_set),
         pred = mu + b_i_reg + b_u_reg + b_r + b_d + b_g) |>
  summarize(rmse = RMSE(rating, pred)) |>
  pull(rmse)
rmse_final_holdout_test