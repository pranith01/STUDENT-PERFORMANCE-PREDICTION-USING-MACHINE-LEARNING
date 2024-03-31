



#Data source downloaded from : https://archive.ics.uci.edu/ml/datasets/student+performance

data <- read.csv("student-por.csv")
dim(data)

str(data)


summary(data)

#Find Missing data: 
sum(is.na(data))

# Split the concatenated string into separate columns
data_split <- strsplit(data[, 1], ";")

# Convert the list to a data frame
data_df <- as.data.frame(do.call(rbind, data_split))

# Rename the columns if necessary
colnames(data_df) <- c("school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", 
                       "reason", "guardian", "traveltime", "studytime", "failures", "schoolsup", "famsup", "paid", 
                       "activities", "nursery", "higher", "internet", "romantic", "famrel", "freetime", "goout", 
                       "Dalc", "Walc", "health", "absences", "G1", "G2", "G3")

# Convert numeric columns to numeric type
numeric_cols <- c("age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout", 
                  "Dalc", "Walc", "health", "absences", "G1", "G2", "G3")

data_df[numeric_cols] <- lapply(data_df[numeric_cols], as.numeric)

# Add the 'passed' column
data_df$passed <- ifelse(data_df$G3 > 9.99, 1, 0)

# Now 'data_df' contains the separated columns and the 'passed' column
str(data_df)

# Get the numeric columns
numeric_cols <- sapply(data_df, is.numeric)

# Extract numeric columns
numeric_data <- data_df[, numeric_cols]

# Number of numeric columns
num_numeric_cols <- sum(numeric_cols)

# Set up the plotting grid
par(mfrow = c(4, ceiling(num_numeric_cols / 4)))

# Plot histograms for numeric variables
for (i in seq_along(numeric_data)) {
  hist(numeric_data[, i], main = names(numeric_data)[i], xlab = names(numeric_data)[i])
}

# Load the imputeMissings package
library(imputeMissings)

# Handle missing values using the impute function
data_clean <- impute(data_df, method = "median")

# Convert data_clean to a data frame if it's not already
if (!is.data.frame(data_clean)) {
  data_clean <- as.data.frame(data_clean)
}

# Convert all columns to numeric
data_clean_numeric <- as.data.frame(lapply(data_clean, as.numeric))

# Check for any NAs introduced during conversion
sum(is.na(data_clean_numeric))

# Compute the correlation matrix
correlation_matrix <- cor(data_clean_numeric)

# Plot the correlation matrix using corrplot
corrplot(correlation_matrix, type = "upper", order = "hclust", sig.level = 0.01, insig = "blank")

# Set up the plot size and margins
par(mar = c(0, 0, 0, 0))  # Adjust the margins to be minimal

corrplot(correlation_matrix, type = "upper", order = "hclust", sig.level = 0.01, insig = "blank", 
         addrect = 4, mar = c(0, 0, 0, 0), outer.margins = c(0, 0, 0, 0),
         width = 8, height = 8)



#Because G1 and G2 are highly correlated with G3, as these are scores students got in two previous classes at the end of the year and we propose to predict value G3 without them, so we decided to remove G1&G2 from our model.
student_data<-dplyr::select(data_df, -c(G1, G2))


#Change the nominal variables to binary variables
student_data<-fastDummies::dummy_cols(student_data, select_columns = c("goout","Dalc", "Walc"), 
                                      remove_most_frequent_dummy = T, remove_selected_columns = T)

#Splitting data on train and test set
#create train and test set
set.seed(333)

trainingRowIndex <- sample(1:nrow(student_data), 0.75*nrow(student_data))  # row indices for training data
train_data <- student_data[trainingRowIndex, ]  # model training data
dim(train_data)

with(train_data, table(passed, useNA = "always"))

test_data  <- student_data[-trainingRowIndex, ]   # model test data
dim(test_data)  


with(test_data, table(passed, useNA = "always"))


#Train a decision tree model:

blr1 <- glm(passed ~ goout_1+goout_2+goout_4+goout_5+Dalc_2+Dalc_3+Dalc_4+Dalc_5+Walc_2+Walc_3+Walc_4+Walc_5, 
            data=train_data, family = "binomial")

summary(blr1) 

car::vif(blr1)

#Test model
#Make predictions on testing data, using trained model:

test_data$blr1.pred <- predict(blr1, newdata = test_data, type = 'response')


# Make confusion matrix:

ProbabilityCutoff <- 0.5  
test_data$blr1.pred.probs <- 1-test_data$blr1.pred

test_data$blr1.pred.passed <- ifelse(test_data$blr1.pred > ProbabilityCutoff, 1, 0)

(cm1 <- with(test_data,table(blr1.pred.passed,passed)))

#Evaluate the model by calculating the accuracy:

CorrectPredictions1 <- cm1[1,1] + cm1[2,2]
TotalStudents1 <- nrow(test_data)

(Accuracy1 <- CorrectPredictions1/TotalStudents1)


# Load the required package
library(rpart)

# Train a decision tree model
tree1 <- rpart(G3 ~ age + Medu + Fedu + traveltime + studytime + failures + famrel + freetime + health + absences +
                 passed + goout_1 + goout_2 + goout_4 + goout_5 + Dalc_2 + Dalc_3 + Dalc_4 + Dalc_5 + Walc_2 + Walc_3 + 
                 Walc_4 + Walc_5, data = train_data, method = 'anova')

summary(tree1)


# Load the required package
library(rattle)

# Train a decision tree model
tree1 <- rpart(G3 ~ age + Medu + Fedu + traveltime + studytime + failures + famrel + freetime + health +
                 absences + passed + goout_1 + goout_2 + goout_4 + goout_5 + Dalc_2 + Dalc_3 + Dalc_4 + 
                 + Walc_2 + Walc_3 + Walc_4 + Walc_5, data = train_data, method = 'anova')


# Set the size of the plotting device
options(repr.plot.width = 12, repr.plot.height = 10) 

# Adjust figure margins
par(mar = c(1, 1, 1, 1))  # Adjust the margins as needed

# Set the size of the plotting device
options(repr.plot.width = 10, repr.plot.height = 8)  # Adjust width and height as needed

# Plot the decision tree
fancyRpartPlot(tree1, caption = "Classification Tree")

# Make predictions on testing data, using trained model:
test_data$tree1.pred.G3 <- predict(tree1, newdata = test_data)

# Visualize predictions:
with(test_data, plot(G3, tree1.pred.G3, main="Actual vs Predicted, testing data", xlab = "Actual G3", ylab = "Predicted G3"))

# Make confusion matrix:

PredictionCutoff <- 10.99 # Use ROC curve to decide cutoff value (this case:in 9-11 range)

test_data$tree1.pred.passed <- ifelse(test_data$tree1.pred.G3 > PredictionCutoff, 1, 0)

(cm2 <- with(test_data, table(tree1.pred.passed, passed)))

# Evaluate the model by calculating the accuracy:

CorrectPredictions2 <- cm2[1,1] + cm2[2,2]
TotalStudents2 <- nrow(test_data)

(Accuracy2 <- CorrectPredictions2 / TotalStudents2)

# Final output dataset for visual validation, using predicted results:
predicted_data <- dplyr::select(test_data, -c(goout_1, goout_2, goout_4, goout_5,
                                              Dalc_2, Dalc_3, Dalc_4, Dalc_5, 
                                              Walc_2, Walc_3, Walc_4, Walc_5,
                                              blr1.pred, blr1.pred.probs, tree1.pred.passed))




