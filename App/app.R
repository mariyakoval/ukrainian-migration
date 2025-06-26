library(shiny)
library(ggplot2)
library(randomForest)
library(caret)
library(pROC)
library(rms)
library(shinydashboard)

ui <- dashboardPage(
  dashboardHeader(title = "Employment of Ukrainian Refugees",
                  titleWidth = 450),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Variable Importance (RF)", tabName = "importance", icon = icon("chart-bar")),
      menuItem("ROC Curve", tabName = "roc", icon = icon("chart-area")),
      menuItem("Logistic Regression Summary", tabName = "logit", icon = icon("file-alt")),
      menuItem("Predict Employment", tabName = "predict", icon = icon("user-check"))
    )
  ),
  
  dashboardBody(
    selectizeInput(
      inputId = "selected_predictors",
      label = "Select Predictors for Model and Prediction:",
      choices = predictors,  # full list of all predictors you have
      selected = predictors, # default: all selected
      multiple = TRUE,
      options = list(plugins = list('remove_button'))
    ),
    fluidRow(
      column(12,
             selectInput("country", "Select Country:",
                         choices = unique(combined_data$country)))
    ),
    
    tabItems(
      tabItem(tabName = "importance",
              fluidRow(
                column(12, h2("Random Forest Variable Importance"), align = "center")
              ),
              fluidRow(
                column(12, plotOutput("varImpPlot"))
              )
      ),
      
      tabItem(tabName = "roc",
              fluidRow(
                column(12, h2("Model ROC Curve"), align = "center")
              ),
              fluidRow(
                column(12, plotOutput("rocPlot"))
              )
      ),
      
      tabItem(tabName = "logit",
              fluidRow(
                column(12, h2("Logistic Regression Model Summary"), align = "center")
              ),
              fluidRow(
                column(12,
                       p("Below is the regression output based on selected  and country data.",
                         style = "font-size: 13px"),
                       verbatimTextOutput("modelSummary"))
              )
      ),
      
      tabItem(tabName = "predict",
              fluidRow(
                column(12, h2("Predict Employment Probability"), align = "center")
              ),
              fluidRow(
                column(12,
                       p("Adjust the variables below to simulate employment outcomes.",
                         style = "font-size: 13px"))
              ),
              fluidRow(
                column(12,
                       uiOutput("dynamicInputs"),
                       actionButton("predictButton", "Predict Employment", icon = icon("play")))
              ),
              br(),
              fluidRow(
                column(12,
                       verbatimTextOutput("predictionResult"))
              )
      )
    )
  )
  )


#-------------------------------------------------------------------------------

server <- function(input, output, session) {

  selected_preds <- reactive({
    req(input$selected_predictors)
    input$selected_predictors
  })
  
  output$dynamicInputs <- renderUI({
    req(selected_preds(), model_data())
    
    tagList(
      lapply(selected_preds(), function(var) {
        values <- sort(unique(model_data()[[var]]))
        
        if (is.factor(model_data()[[var]]) || is.character(model_data()[[var]]) || length(values) <= 5) {
          selectInput(inputId = var,
                      label = paste("Select:", var),
                      choices = values,
                      selected = values[1])
        } else {
          numericInput(inputId = var,
                       label = paste("Enter value for:", var),
                       value = round(mean(as.numeric(values), na.rm = TRUE), 2))
        }
      })
    )
  })
  
  output$predictionResult <- renderPrint({
    req(input$predictButton)
    
    isolate({
      df <- model_data()
      user_inputs <- sapply(input$selected_predictors, function(var) input[[var]], simplify = FALSE)
      user_df <- as.data.frame(user_inputs, stringsAsFactors = FALSE)
      
      for (var in input$selected_predictors) {
        if (is.factor(df[[var]])) {
          user_df[[var]] <- factor(user_df[[var]], levels = levels(df[[var]]))
        } else {
          user_df[[var]] <- as.numeric(user_df[[var]])
        }
      }
      
      pred <- predict(logit_model(), newdata = user_df, type = "response")
      paste("Predicted probability of employment:", round(pred, 3))
    })
  })
  
  
  
  model_data <- reactive({
    req(selected_preds())
    df <- combined_data %>%
      filter(country == input$country) %>%
      select(employed_binary, all_of(input$selected_predictors)) %>%
      drop_na()
    df$introduction_resp_age <- factor(df$introduction_resp_age)
    df$demographics_educ_level_grouped <- factor(df$demographics_educ_level_grouped)
    df$employed_binary <- factor(df$employed_binary, levels = c(0, 1), labels = c("No", "Yes"))
    return(df)
  })
  var_classes <- reactive({
    sapply(model_data()[, input$selected_predictors, drop = FALSE], class)
  })
  
  rf_model <- reactive({
    req(selected_preds())
    randomForest(as.formula(paste("employed_binary ~", paste(selected_preds(), collapse = "+"))),
                 data = model_data(),
                 ntree = 500,
                 importance = TRUE)
  })
  
  logit_model <- reactive({
    req(selected_preds())
    glm(as.formula(paste("employed_binary ~", paste(selected_preds(), collapse = "+"))),
        data = model_data(),
        family = binomial)
  })
  
  output$varImpPlot <- renderPlot({
    varImpPlot(rf_model())
  })
  
  output$rocPlot <- renderPlot({
    probs <- predict(logit_model(), type = "response")
    roc_obj <- roc(model_data()$employed_binary, probs)
    plot(roc_obj, main = "ROC Curve")
  })
  
  output$modelSummary <- renderPrint({
    summary(logit_model())
  })
  
  
  output$predictionResult <- renderPrint({
    req(input$predictButton, selected_preds())
    
    isolate({
      df <- model_data()
      user_inputs <- sapply(selected_preds(), function(var) input[[var]], simplify = FALSE)
      user_df <- as.data.frame(user_inputs, stringsAsFactors = FALSE)
      
      for (var in selected_preds()) {
        if (is.factor(df[[var]]) || is.character(df[[var]])) {
          user_df[[var]] <- factor(user_df[[var]], levels = levels(df[[var]]))
        } else {
          user_df[[var]] <- as.numeric(user_df[[var]])
        }
      }
      
      pred <- predict(logit_model(), newdata = user_df, type = "response")
      paste("Predicted probability of employment:", round(pred, 3))
    })
  })
  
}

#---------------

shinyApp(ui = ui, server = server)