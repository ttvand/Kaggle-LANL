# Define UI
shinyUI(navbarPage(
  "Exploratory data analysis",
  theme = shinytheme("cerulean"), #united
  tabPanel("Train data analysis",
           sidebarPanel(
             sliderInput("data_start", "Data selection", 0, 1-increment_period,
                         init_rand_start, increment_period),
             fluidRow(
               column(2, actionButton("prev_period", "",
                                      icon("arrow-alt-circle-left"))),
               column(2, actionButton("next_period", "",
                                      icon("arrow-alt-circle-right")))
             ),
             width = 3),
           mainPanel(
             br(),
             plotlyOutput("quake_train_plotly")
           )
  ),
  tabPanel("Test data analysis",
           sidebarPanel(
             selectInput("test_file", "Test file name", test_files,
                         init_test_file),
             fluidRow(
               column(2, actionButton("prev_test", "",
                                      icon("arrow-alt-circle-left"))),
               column(2, actionButton("next_test", "",
                                      icon("arrow-alt-circle-right")))
             ),
             width = 3),
           mainPanel(
             br(),
             plotlyOutput("quake_test_plotly")
           )
  )
))

