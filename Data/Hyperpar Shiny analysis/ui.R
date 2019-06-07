# Define UI
shinyUI(navbarPage(
  "Hyperparameter performance analysis",
  theme = shinytheme("cerulean"), #united
  tabPanel("Aggregate performance boxplots",
           sidebarPanel(
             selectInput("x1", "X1", analysis_cols_first,
                         selected=first_selected),
             fluidRow(
               column(2, actionButton("prev_x1", "",
                                      icon("arrow-alt-circle-left"))),
               column(2, actionButton("next_x1", "",
                                      icon("arrow-alt-circle-right")))
             ),
             br(),
             
             selectInput("x2", "X2", analysis_cols_second),
             fluidRow(
               column(2, actionButton("prev_x2", "",
                                      icon("arrow-alt-circle-left"))),
               column(2, actionButton("next_x2", "",
                                      icon("arrow-alt-circle-right")))
             ),
             width = 3),
           mainPanel(
             br(),
             plotlyOutput("boxplot_mae_plotly")
           )
  )
))

