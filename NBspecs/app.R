library(shiny)
library(tm)
library(RTextTools)
library(e1071)
library(dplyr)
library(readr)
library(caret)
# Library for parallel processing
library(doMC)
registerDoMC(cores=detectCores())
library(CORElearn)
library(pROC)
library(fmsb)
library(lsa)
library(DT)

# Define UI for application that draws a histogram
ui <- fluidPage(
   
   # Application title
   titlePanel("Find Best Naive Bayes Classifier"),
   
   # Sidebar with a slider input for number of bins 
   sidebarLayout(
      fluidRow(
        sidebarPanel(width=3,
         sliderInput("Ntrain",
                     "Training sample size:",
                     min = 1,
                     max = 3000,
                     value = 1000),
         sliderInput("minFreq",
                     "Minimum frequency in entire copus to be considered as feature:",
                     min = 1,
                     max = 200,
                     value = 50),
         sliderInput("minIG",
                     "Minimum information gain to be selected as feature:",
                     min = 0,
                     max = 0.01,
                     value = 0.001,
                     step = 0.0002),
         actionButton("update", "Update"),
         helpText('Select values and hit "Update" to see results. 
                  Change values and update again to see overlaying radar plot from different runs.')
      ),
      
        mainPanel(
         fluidRow(column(width=8, plotOutput("distPlot")),
                  column(width=4, 
                         fluidRow(tableOutput("twoBytwo")),
                         fluidRow(tableOutput("confM"))))
         )
      ),
      fluidRow(
        column(width=1),
        column(width=4, textOutput("Info5"),
                        textOutput("Info1"),
                        textOutput("Info2"),
                        textOutput("Info3"),
                        textOutput("Info4")),
        column(width=6, dataTableOutput("training")),
        column(width=1))
   )
)

# Define server logic required to draw a histogram
server <- function(input, output){
  
  Res<-reactiveValues(df = data.frame(Accuracy=c(),
                                      Sensitivity=c(),
                                      Specificity=c(),
                                      Kappa=c(),
                                      PosPred=c(),
                                      NegPred=c()))
  
  observeEvent(input$update,{
  
    withProgress(message = "This will take a few seconds ...", {
    
  recipes<-suppressWarnings(suppressMessages(read_csv("recipes.csv")))
  recipes<-filter(recipes, !duplicated(recipes$instructions)) %>%
    filter(review_count!=0) %>%
    #https://fulmicoton.com/posts/bayesian_rating/
    #C=median(review_count)=20, m=3
    mutate(rating_b=(20*3+review_count*rating_stars)/(20+review_count),
           four_star=ifelse(rating_b>=3.5, "Y", "N"))
  
  recipes$four_star<-factor(recipes$four_star, levels=c("Y", "N"))
  
  #cleaning special characters from scraping
  #some other cleaning are done in excel spread sheet
  temp<-as.data.frame(recipes)[recipes$title=="Herbed Pecan Crusted Scallops","ingredients"]
  recipes[recipes$title=="Herbed Pecan Crusted Scallops","ingredients"]<-gsub("\xeb\xe5", "", temp)
  recipes<-recipes[c(-1480), ]
  temp<-as.data.frame(recipes)[,"ingredients"]
  recipes[,"ingredients"]<-gsub("\x8c\xac", "", temp)
  
  #add ID var
  recipes$ID<-1:nrow(recipes)
  
  #merge ingredients and instructions to get a corpus for each recipe
  recipes$text<- with(recipes, paste0(ingredients, " ", instructions, " ", title))
  recipes<-select(recipes, c(-ingredients, -instructions))
   corpus <- Corpus(VectorSource(recipes$text))

   #get rid of punctuaiton, numbers, whitespaces, ect.
   corpus_clean <- 
     corpus %>%
     tm_map(content_transformer(tolower)) %>% 
     tm_map(removePunctuation) %>%
     tm_map(removeNumbers) %>%
     tm_map(removeWords, stopwords(kind="en")) %>%
     tm_map(stripWhitespace)
   
   #document term matrix
   dtm <- DocumentTermMatrix(corpus_clean)
   
   #training and testing
   #creat id data for later merging corpus with other covariates
   train<-sample(recipes$ID, input$Ntrain, replace=FALSE)
   test<-recipes$ID[!recipes$ID %in% train]
   
   recipes.train <- recipes[train,]
   id.train <- data.frame(ID=recipes.train$ID)
   recipes.test <- recipes[test,]
   id.test <- data.frame(ID=recipes.test$ID)
   
   dtm.train <- dtm[train,]
   dtm.test <- dtm[test,]
   
   corpus_clean.train <- corpus_clean[train]
   corpus_clean.test <- corpus_clean[test]
   
   #dim(dtm.train)
   freq <- findFreqTerms(dtm.train, input$minFreq)
   #length((freq))
   
   dtm.train<- DocumentTermMatrix(corpus_clean.train, control=list(dictionary = freq))
   dtm.test<- DocumentTermMatrix(corpus_clean.test, control=list(dictionary = freq))
   
   #convert term features to "Yes" or "No"
   convert_count <- function(x) {
     y <- ifelse(x > 0, 1,0)
     y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
     y
   }
   
   trainNB <- apply(dtm.train, 2, convert_count)
   id.train<-left_join(id.train, recipes.train[, c("ID", "total_time", "prep_time", "cook_time", "four_star")], by="ID") 
   trainNB <- cbind(trainNB, id.train)
   trainNB$four_star<-as.factor(trainNB$four_star)
   
   testNB <- apply(dtm.test, 2, convert_count)
   id.test<-left_join(id.test, recipes.test[, c("ID", "total_time", "prep_time", "cook_time", "four_star")], by="ID")
   testNB <- cbind(testNB, id.test)
   
   
   #feature selection using information gain
   IG.CORElearn <- attrEval(four_star ~ ., data=trainNB,  estimator = "InfGain")
   features<-IG.CORElearn[IG.CORElearn>=input$minIG]
   
   trainNB.sub<-trainNB[,names(trainNB) %in% names(features)] %>% select(-ID)
   testNB.sub<-testNB[,names(testNB) %in% names(features)] %>% select(-ID)
   
   #train classifier and predict
   system.time( classifier <- naiveBayes(trainNB.sub, recipes.train$four_star, laplace = 0) )
   system.time( pred <- predict(classifier, newdata=testNB.sub) )
   
   tb<-table("Predictions"= pred,  "Actual" = recipes.test$four_star)
   # twobytwo<-as.data.frame(matrix(c(tb), nrow=2, byrow=TRUE))
   # twobytwo<-rbind(c("Y", "N"), twobytwo)
   # twobytwo$Pre<-c("Pred", "Y", "N")
   # twobytwo<-twobytwo[,c(3,1,2)]
   # names(twobytwo)<-c("", "Actual", "")
   
   conf.mat <- confusionMatrix(pred, recipes.test$four_star, positive="Y")
   
   maxmin <- data.frame(
     Accuracy=c(1, 0),
     Sensitivity=c(1, 0),
     Specificity=c(1, 0),
     Kappa=c(1, 0),
     PosPred=c(1, 0),
     NegPred=c(1, 0))
   
   res <- data.frame(
     Accuracy=conf.mat$overall['Accuracy'],
     Sensitivity=conf.mat$byClass['Sensitivity'],
     Specificity=conf.mat$byClass['Specificity'],
     Kappa=conf.mat$overall['Kappa'],
     PosPred=conf.mat$byClass['Pos Pred Value'],
     NegPred=conf.mat$byClass['Neg Pred Value'])
   
   Res$df<-rbind(Res$df, res)
   
   res_out<-cbind(c('Accuracy', 'Sensitivity', 'Specificity', 'Kappa', 'Pos Pred Value', 'Neg Pred Value'), 
                  round(t(res),3))
   colnames(res_out)<-c("Metric", "Value")
   
   RNGkind("Mersenne-Twister")
   #dat <- rbind(maxmin,res)
   
   
     output$distPlot <- renderPlot({
       dat <- rbind(maxmin,Res$df)
       radarchart(dat, axistype=3, pty=32, plty=1, axislabcol="white", na.itp=FALSE,
                  vlabels=c("Accuracy", "Sensitivity", "Specificity", "Kappa", "Pos Pred Value", "Neg Pred Value"),
                  title="Plot radar", pcol=2:8)
     }, height = 400, width = 600 )
   
   
     output$training<-renderDataTable({
         train_out<-cbind(recipes.train$title, as.data.frame(trainNB.sub))
         names(train_out)[1]<-"recipe name"
         head(train_out, 20)
     }, options = list(pageLength = 200, scrollX = TRUE, scrollY = "300px"))
   
     output$twoBytwo<-renderTable({
         tb
     })
   
     output$confM<-renderTable({
       res_out
     })
     
     output$Info1<-renderPrint({
       cat('Dimension of original dataset "recipes": ', dim(recipes)[1], ',', dim(recipes)[2])
     })
     
     output$Info2<-renderPrint({
       cat("Numbers of features in total: ", dim(dtm)[2])
     })
     
     output$Info3<-renderPrint({
       cat("Number of features used in training: ", dim(trainNB.sub)[2])
     })
     
     output$Info4<-renderPrint({
       cat('Features used in training: ', paste0('"', names(trainNB.sub), '"'))
     })
     
     output$Info5<-renderPrint({
       cat("The first 20 rows of the training data is displyed on the right.")
     })
     
  })#ends withProgress
  })#ends observeEvent
    
  
   
}#ends server

# Run the application 
shinyApp(ui = ui, server = server)

