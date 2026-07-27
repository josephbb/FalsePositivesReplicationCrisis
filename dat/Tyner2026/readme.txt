This folder contains the following files:
“analysis and knit.R” contains all the code to generate all stats and figures and to knit it into the placeholder manuscript document
“analyst data.RData” contains all datasets needed to generate the stats and figures
“common functions.R” contains helper functions for statistics and figures
“repli_binary.R” contains optional code pertaining to the binary replication measures
“template manuscript.docx” is the full manuscript file, where all statistics and figures are placeholders to be calculated
"Analyst package.Rproj" is an R Project file for setting the root directory
 
The template manuscript.docx contains a version of the manuscript with tagged statistics and text contained in {} brackets.
 
To reproduce those tagged statistics:
 
Preparation (only needs to be done once):
1) Extract or download all files noted above into the same folder
2) (Optional) Install R and R studio
3) Open the "Analyst package.Rproj" to ensure all files run from the correct directory
4) Run the following code to install the renv package if not already installed:
install.packages("renv"))
renv::init()
5) When prompted with "This project already has a lockfile. What would you like to do?" Select "1: Restore the project from the lockfile."
Please note that it is possible that some external dependencies will need to be installed manually, depending on your system.
6) Restart R

Generating tagged statistics and figures:
1) Open the "Analyst package.Rproj" to ensure all files run from the correct directory
2) Run the "analysis and knit.R" file to prepare all functions needed
3) In the console, knit_manuscript() function
4) This will do as follows:
  a) Load all data and functions needed
  b) Generate the full set of statistics and figures into the R environment as two objects:
     i) "results_placeholder_stats" containing all the statistics contained within the manuscript, and 
     iii) "results_figures" containing all the figures as ggplot objects
  c) Search the template manuscript document for placeholders contained in {} brackets
  d) Replace all placeholders with calculated text and figures (i.e. "knit" the manuscript document) and create/replace the "knitted manuscript.docx" file
5) To access those statistics and figures in the global environment, simply locate them with $, as in results_placeholder_stats$n_claims or results_figures$figure_2
 
Tested with:
R version 4.5.0 (2024-10-31)
Platform: aarch64-apple-darwin20
Running under: macOS Sequoia 15.3.2

Microsoft Word is recommended for viewing the output manuscript document .docx file due to potential issues with rendering.
 
No non-standard hardware required
 
Install time from scratch: 20 minutes
Typical run time: 5 minutes
Expected output: R objects containing all figures and statistics
 
Package versions as tested:
- abind                [* -> 1.4-8]
- askpass              [* -> 1.2.1]
- assertthat           [* -> 0.2.1]
- backports            [* -> 1.5.0]
- base64enc            [* -> 0.1-3]
- BayesFactor          [* -> 0.9.12-4.7]
- BayesRep             [* -> 0.42.2]
- bayestestR           [* -> 0.17.0]
- BFF                  [* -> 4.4.2]
- BH                   [* -> 1.87.0-1]
- bit                  [* -> 4.6.0]
- bit64                [* -> 4.6.0-1]
- boot                 [* -> 1.3-32]
- bridgesampling       [* -> 1.1-2]
- Brobdingnag          [* -> 1.2-9]
- broom                [* -> 1.0.10]
- bslib                [* -> 0.9.0]
- cachem               [* -> 1.1.0]
- callr                [* -> 3.7.6]
- checkmate            [* -> 2.3.3]
- cli                  [* -> 3.6.5]
- clipr                [* -> 0.8.0]
- cluster              [* -> 2.1.8.1]
- coda                 [* -> 0.19-4.1]
- codetools            [* -> 0.2-20]
- colorspace           [* -> 2.1-2]
- colourpicker         [* -> 1.3.0]
- commonmark           [* -> 2.0.0]
- contfrac             [* -> 1.1-12]
- corrplot             [* -> 0.95]
- cowplot              [* -> 1.2.0]
- cpp11                [* -> 0.5.2]
- crayon               [* -> 1.5.3]
- crosstalk            [* -> 1.2.2]
- curl                 [* -> 7.0.0]
- data.table           [* -> 1.17.8]
- datawizard           [* -> 1.3.0]
- desc                 [* -> 1.4.3]
- deSolve              [* -> 1.40]
- digest               [* -> 0.6.37]
- distributional       [* -> 0.5.0]
- dplyr                [* -> 1.1.4]
- DT                   [* -> 0.34.0]
- effectsize           [* -> 1.0.1]
- elliptic             [* -> 1.5-0]
- evaluate             [* -> 1.0.5]
- farver               [* -> 2.1.2]
- fastmap              [* -> 1.2.0]
- fontawesome          [* -> 0.5.3]
- forcats              [* -> 1.0.1]
- foreach              [* -> 1.5.2]
- foreign              [* -> 0.8-90]
- Formula              [* -> 1.2-5]
- fs                   [* -> 1.6.6]
- funkyheatmap         [* -> 0.5.2]
- gargle               [* -> 1.6.0]
- gdata                [* -> 3.0.1]
- generics             [* -> 0.1.4]
- ggExtra              [* -> 0.11.0]
- ggforce              [* -> 0.5.0]
- ggplot2              [* -> 4.0.0]
- ggridges             [* -> 0.5.7]
- ggside               [* -> 0.4.0]
- glmnet               [* -> 4.1-10]
- glue                 [* -> 1.8.0]
- googledrive          [* -> 2.1.2]
- gridExtra            [* -> 2.3]
- gsl                  [* -> 2.1-8]
- gtable               [* -> 0.3.6]
- gtools               [* -> 3.9.5]
- haven                [* -> 2.5.5]
- highr                [* -> 0.11]
- Hmisc                [* -> 5.2-4]
- hms                  [* -> 1.1.4]
- htmlTable            [* -> 2.4.3]
- htmltools            [* -> 0.5.8.1]
- htmlwidgets          [* -> 1.6.4]
- httpuv               [* -> 1.6.16]
- httr                 [* -> 1.4.7]
- hypergeo             [* -> 1.2-14]
- inline               [* -> 0.3.21]
- insight              [* -> 1.4.2]
- isoband              [* -> 0.2.7]
- iterators            [* -> 1.0.14]
- jomo                 [* -> 2.7-6]
- jquerylib            [* -> 0.1.4]
- jsonlite             [* -> 2.0.0]
- knitr                [* -> 1.50]
- labeling             [* -> 0.4.3]
- lamW                 [* -> 2.2.5]
- LaplacesDemon        [* -> 16.1.6]
- later                [* -> 1.4.4]
- lattice              [* -> 0.22-7]
- lazyeval             [* -> 0.2.2]
- lifecycle            [* -> 1.0.4]
- lme4                 [* -> 1.1-37]
- logspline            [* -> 2.1.22]
- loo                  [* -> 2.8.0]
- magrittr             [* -> 2.0.4]
- MASS                 [* -> 7.3-65]
- mathjaxr             [* -> 1.8-0]
- Matrix               [* -> 1.7-4]
- MatrixModels         [* -> 0.5-4]
- matrixStats          [* -> 1.5.0]
- memoise              [* -> 2.0.1]
- metaBMA              [* -> 0.6.9]
- metadat              [* -> 1.4-0]
- metafor              [* -> 4.8-0]
- mice                 [* -> 3.18.0]
- mime                 [* -> 0.13]
- miniUI               [* -> 0.1.2]
- minqa                [* -> 1.2.8]
- mitml                [* -> 0.4-5]
- mnormt               [* -> 2.1.1]
- mvtnorm              [* -> 1.3-3]
- nlme                 [* -> 3.1-168]
- nloptr               [* -> 2.2.1]
- nnet                 [* -> 7.3-20]
- numDeriv             [* -> 2016.8-1.1]
- officer              [* -> 0.7.0]
- openssl              [* -> 2.3.4]
- ordinal              [* -> 2023.12-4.1]
- otel                 [* -> 0.2.0]
- pan                  [* -> 1.9]
- pandoc               [* -> 0.2.0]
- parameters           [* -> 0.28.2]
- patchwork            [* -> 1.3.2]
- pbapply              [* -> 1.7-4]
- performance          [* -> 0.15.2]
- pillar               [* -> 1.11.1]
- pkgbuild             [* -> 1.4.8]
- pkgconfig            [* -> 2.0.3]
- polyclip             [* -> 1.10-7]
- posterior            [* -> 1.6.1]
- prettyunits          [* -> 1.2.0]
- processx             [* -> 3.8.6]
- progress             [* -> 1.2.3]
- promises             [* -> 1.4.0]
- ps                   [* -> 1.9.1]
- purrr                [* -> 1.1.0]
- pwr                  [* -> 1.3-0]
- QuickJSR             [* -> 1.8.1]
- R6                   [* -> 2.6.1]
- ragg                 [* -> 1.5.0]
- rappdirs             [* -> 0.3.3]
- rbibutils            [* -> 2.3]
- RColorBrewer         [* -> 1.1-3]
- Rcpp                 [* -> 1.1.0]
- RcppArmadillo        [* -> 15.0.2-2]
- RcppEigen            [* -> 0.3.4.0.2]
- RcppParallel         [* -> 5.1.11-1]
- Rdpack               [* -> 2.6.4]
- readr                [* -> 2.1.5]
- reformulas           [* -> 0.4.1]
- renv                 [* -> 1.1.5]
- ReplicationSuccess   [* -> 1.3.3]
- rlang                [* -> 1.1.6]
- rmarkdown            [* -> 2.30]
- rpart                [* -> 4.1.24]
- rstan                [* -> 2.32.7]
- rstantools           [* -> 2.5.0]
- rstudioapi           [* -> 0.17.1]
- S7                   [* -> 0.2.0]
- sass                 [* -> 0.4.10]
- scales               [* -> 1.4.0]
- shape                [* -> 1.4.6.1]
- shiny                [* -> 1.11.1]
- shinyjs              [* -> 2.1.0]
- sourcetools          [* -> 0.1.7-1]
- StanHeaders          [* -> 2.32.10]
- stringi              [* -> 1.8.7]
- stringr              [* -> 1.5.2]
- survival             [* -> 3.8-3]
- sys                  [* -> 3.4.3]
- systemfonts          [* -> 1.3.1]
- tensorA              [* -> 0.36.2.1]
- textshaping          [* -> 1.0.4]
- tibble               [* -> 3.3.0]
- tidyr                [* -> 1.3.1]
- tidyselect           [* -> 1.2.1]
- tinytex              [* -> 0.57]
- tweenr               [* -> 2.0.3]
- tzdb                 [* -> 0.5.0]
- ucminf               [* -> 1.2.2]
- utf8                 [* -> 1.2.6]
- uuid                 [* -> 1.2-1]
- vctrs                [* -> 0.6.5]
- viridisLite          [* -> 0.4.2]
- vroom                [* -> 1.6.6]
- wCorr                [* -> 1.9.8]
- weights              [* -> 1.1.2]
- withr                [* -> 3.0.2]
- xfun                 [* -> 0.53]
- xml2                 [* -> 1.4.1]
- xtable               [* -> 1.8-4]
- yaml                 [* -> 2.3.10]
- zip                  [* -> 2.3.3]

 
The MIT License (MIT)
Copyright (c) 2025 Center for Open Science
 
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.