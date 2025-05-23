library(fpp2)
library(forecast)
library(rugarch)

data <- read.csv('../Downloads/data_up.csv', sep=';')

vett_prices <- as.vector(t(c(data[,2:25])))
vett_volumes <- c(data[,26:49])

###
factor <- '.'
for (i in 1:8030){
  if (nchar(as.character(vett_prices[[2]])[i])>6){
    factor <- as.character(vett_prices[[2]])[i]
    counter <- 0
    j <- 1
    while (counter<2){
      if (substr(factor, start=j, stop=j)=='.'){
        counter = counter+1
      }
      j <- j+1
    }
    levels(vett_prices[[2]]) <- c(levels(vett_prices[[2]]), substr(factor, start=1, stop=j-2))
    vett_prices[[2]][[i]] <- substr(factor, start=1, stop=j-2)
    print(vett_prices[[2]][[i]])
  }
}

numeric_vec <- as.numeric(as.character(vett_prices[[2]]))
vett_prices[[2]] <- numeric_vec
###

matrix_prices <- t(do.call(cbind, vett_prices))
matrix_volumes <- t(do.call(cbind, vett_volumes))

save(matrix_prices, file = "matrix_prices.RData")

scalar_prices <- c(matrix_prices)
print(scalar_prices[12:28])
scalar_volumes <- c(matrix_volumes)
print(scalar_volumes[12:28])

st_prices <- ts(scalar_prices)
window_prices <- window(st_prices, end=24*731)

st_volumes <- ts(scalar_volumes)
window_volumes <- window(st_volumes, start=1, end=24*731)

dates <- seq(from = as.POSIXct("2000-01-01 01:00:00", format = "%Y-%m-%d %H:%M:%S"),
             to = as.POSIXct("2002-01-01 00:00:00", format = "%Y-%m-%d %H:%M:%S"),
             by = "hour")

df_dates <- data.frame(colonna1=c(dates), colonna2=window_prices, colonna3=window_volumes)
st_prices <- ts(df_dates[,2], start=c(2000,1), end=c(2002,1), frequency=24*365)
st_volumes <- ts(df_dates[,3], start=c(2000,1), end=c(2002,1), frequency=24*365)

st_prices <- window(st_prices, start=c(2001,1), end=c(2002,1), frequency=24*365)
st_volumes <- window(st_volumes, start=c(2001,1), end=c(2002,1), frequency=24*365)

#correlation between variables
df <- data.frame(colonna1=scalar_prices, colonna2=scalar_volumes)

GGally::ggpairs(as.data.frame(df[1:10000,]))

#for (k in 1:8000){
#  if (scalar_prices[k]>800){
#    print(k)
#  }
#}

diff_prices <- st_prices[2:length(st_prices)]-st_prices[1:(length(st_prices)-1)]
diff_prices <- ts(diff_prices, start=c(2001,2), frequency=365*24)
#plot
p1 <- autoplot(st_prices) +
  ggtitle("") +
  xlab("time (hours)") + ylab("prices") +
  theme_bw() +
  scale_x_continuous(breaks=seq(2001,2002,by=1/(12*3)), labels=seq(from = as.Date("2001/01/01"),
                                                                    to = as.Date("2001/12/27"),
                                                                    by = "10 days")) +
  #ylim(-250,1000) +
  guides(colour=guide_legend(title="price"))
p2 <- autoplot(diff_prices) +
  ggtitle("DAM diff prices") +
  xlab("date") + ylab("diff prices") +
  theme_bw() +
  guides(colour=guide_legend(title="diff prices"))
gridExtra::grid.arrange(p1, p2, ncol = 1)

#
# SES
#
h <- 1752 #20%
t <- 7009 #80% train
ses_model <- ses(window(st_prices,end=c(2001,t), frequency=365*24), h=1)
ses_model
alpha <- as.numeric(ses_model$model$par[1])
round(accuracy(ses_model),3)
res_ses_train <- ses_model$residuals

#ses forecasting
ses_train <- fitted(ses_model)
#h <- 4381
#t <- 13140
ses_test <- ts(rep(0,h))
sum <- as.numeric(ses_model$model$par[2])*(1-alpha)^t
for (j in 1:t){
  sum <- sum + st_prices[j]*alpha*(1-alpha)^(t-j)
}
ses_test[1] <- sum
for (i in 1:(h-1)){
  ses_test[i+1] <- ses_test[i]*(1-alpha) + alpha*st_prices[t+i]
}
ses_test <- ts(ses_test,start=c(2001,t+1),frequency=365*24)
ses_train <- ts(ses_train,start=c(2001,2), end=c(2001,t),frequency=365*24)
#calcolo gli errori di forecast ed mse
forecast_ses <- window(st_prices,start=c(2001,t+1))-ses_test
mse_ses <- mean(forecast_ses^2)
#creo un unico vettore per poterlo plottare
ses_prices<-ts(c(as.vector(ses_train), as.vector(ses_test)),
               start=c(2001,2), end=c(2002,1), frequency=365*24)
res_ses <- ts(c(as.vector(res_ses_train), as.vector(forecast_ses)),
              start=c(2001,2), end=c(2002,1), frequency=365*24)

#ses plot
models_ses <- ts(data.frame(y1=window(st_prices,start=c(2001,2)),
                            y2=ses_prices), start=c(2001,2), end=c(2002,1), frequency=365*24)
p1<-autoplot(models_ses) +
  labs(title = paste("Simple Exponential Smoothing: MSE_train = ", round(ses_model$model$mse,6), ", MSE_test = ", round(mse_ses,6)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "SES")) +
  geom_vline(xintercept=2001.8, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(res_ses) +
  labs(x = "years", y = "prices", title = "Residuals for DAM prices") +
  scale_color_manual(values = c("SES              " = "blue")) +
  geom_line(aes(color = "SES              ")) +  # Aggiungo una mappatura del colore
  geom_vline(xintercept=2001.8, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#
# naive
#
naive_train <- fitted(naive(window(st_prices, end=c(2001,t), frequency=365*24)))
#naive forecasting
naive_test <- ts(rep(0,h))
for (i in 1:h){
  naive_test[i] <- st_prices[t+i-1]
}
#naive: forecast error, mse
forecast_naive <- window(st_prices,start=c(2001,t+1),frequency=365*24)-ts(naive_test,start=c(2001,t+1),frequency=365*24)
mse_naive_test <- mean(forecast_naive^2)
#naive: residual
res_naive_train <- window(window(st_prices, start=c(2001,2), end=c(2001,t), frequency=365*24)-naive_train,
                          start=c(2001,2), frequency=365*24)
#naive: fitted accuracy
mse_naive_train <- mean(res_naive_train^2)
#creo un unico vettore per poterlo plottare
naive_train <- window(naive_train, start=c(2001,2))
naive_test <- ts(naive_test, start=c(2001,t+1), frequency=365*24)
st_naive<-ts(c(as.vector(naive_train), as.vector(naive_test)),
                 start=c(2001,2), end=c(2002,1), frequency=365*24)
res_naive <- ts(c(as.vector(res_naive_train), as.vector(forecast_naive)),
                start=c(2001,2), end=c(2002,1), frequency=365*24)
#naive plot
models_naive <- ts(data.frame(y1=window(st_prices, start=c(2001,2), frequency=365*24),
                              y2=st_naive), start=c(2001,2), end=c(2002,1), frequency=365*24)

p1<-autoplot(models_naive) +
  labs(title = paste("Naive Method: MSE_train = ", round(mse_naive_train,6), ", MSE_test = ", round(mse_naive_test,6)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "Naive")) +
  geom_vline(xintercept=2001.8, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(res_naive) +
  labs(x = "years", y = "prices", title = "Residuals for DAM prices") +
  scale_color_manual(values = c("Naive            " = "blue")) +
  geom_line(aes(color = "Naive            ")) +  # Aggiungo una mappatura del colore
  geom_vline(xintercept=2001.8, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#
# ETS
#
ets_model <- ets(window(st_prices ,end=c(2001,t), frequency=365*24),
                 model="ZZZ", damped=NULL, alpha=NULL, beta=NULL,
                 gamma=NULL, phi=NULL, lambda=NULL, biasadj=FALSE,
                 additive.only=FALSE, restrict=TRUE,
                 allow.multiplicative.trend=FALSE)
summary(ets_model)
ets_train <- fitted(ets_model)
#ETS(M,A,N)
b <- ts(rep(0,length(st_prices)))
l <- ts(rep(0,length(st_prices)))
ets_train_fitted <- rep(0,t-1)
ets_test <- ts(rep(0,h))
b[1] <- as.numeric(ets_model$par[4])
l[1] <- as.numeric(ets_model$par[3])
alpha <- as.numeric(ets_model$par[1])
beta <- as.numeric(ets_model$par[2])
for (i in 1:(t-1)){
  l[i+1] <- alpha*st_prices[i]+(1-alpha)*(l[i]+b[i])
  b[i+1] <- b[i]*(1-beta)+beta*(l[i+1]-l[i])
  ets_train_fitted[i+1] <- l[i+1]+b[i+1]
}
ets_train_fitted <- ets_train_fitted[2:t]
ets_train_fitted <- ts(ets_train_fitted, start=c(2001,2),end=c(2001,t), frequency=365*24)
for (j in 1:h){
  l[j+t] <- alpha*st_prices[j+t-1]+(1-alpha)*(l[j+t-1]+b[j+t-1])
  b[j+t] <- b[j+t-1]*(1-beta)+beta*(l[j+t]-l[j+t-1])
  ets_test[j] <- l[j+t]+b[j+t]
}
ets_test <- ts(ets_test, start=c(2001,t+1), frequency=365*24)
res_ets_train <- ts(st_prices[2:t] - ets_train_fitted,
                    start=c(2001,2), end=c(2001,t), frequency=365*24)
forecast_ets <- ts(st_prices[(t+1):length(st_prices)] - ets_test,
                   start=c(2001,t+1), end=c(2002,1), frequency=365*24)

mse_ets_train <- mean(res_ets_train^2)
mse_ets_test <- mean(forecast_ets^2)

st_ets<-ts(c(as.vector(ets_train_fitted), as.vector(ets_test)),
                 start=c(2001,2), end=c(2002,1), frequency=365*24)
res_ets <- ts(c(as.vector(res_ets_train), as.vector(forecast_ets)),
                start=c(2001,2), end=c(2002,1), frequency=365*24)
#ets plot
models_ets <- ts(data.frame(y1=window(st_prices, start=c(2001,2), frequency=365*24),
                            y2=st_ets), start=c(2001,2), end=c(2002,1), frequency=365*24)

p1<-autoplot(models_ets) +
  labs(title = paste("ETS(M,A,N) Method: alpha = ",round(alpha,5),", beta = ",round(beta,5)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "ETS")) +
  geom_vline(xintercept=2001.8, linetype = "dashed", color = "black") +
  scale_x_continuous(breaks=seq(2001,2002,by=1)) +
  theme_bw() +
  ylim(-250,1000)
#ses plot residuals
p2 <- autoplot(res_ets) +
  labs(x = "years", y = "residuals", title = paste("Residuals for DAM volumes: MSE_train = ", round(mse_ets_train,6), ", MSE_test = ", round(mse_ets_test,6))) +
  scale_color_manual(values = c("ETS               " = "blue")) +
  geom_line(aes(color = "ETS               ")) +  # Aggiungo una mappatura del colore
  geom_vline(xintercept=2001.8, linetype = "dashed", color = "black") +
  theme_bw()
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#compare plot models
st_models <- ts(data.frame(y1=window(st_prices, start=c(2001,2), frequency=365*24),
                           y2=st_naive,
                           y3=ses_prices,
                           y4=st_ets),
                start=c(2001,2), frequency=365*24)
residuals <- ts(data.frame(y1=res_naive,
                           y2=res_ses,
                           y3=res_ets),
                start=c(2001,2), frequency=365*24)
p1 <- autoplot(st_models) +
  labs(x = "years", y = "prices", title = "Forecasts for DAM prices") +
  scale_color_manual(values = c("black", "blue", "green", "red"),
                     labels = c("Ground truth", "Naive", "SES", "ETS")) +
  theme_bw() +
  geom_vline(xintercept=2001.8, linetype = "dashed", color = "black")
print(p1)
#ses plot residuals
p2 <- autoplot(residuals) +
  labs(x = "years", y = "residuals", title = "Residuals") +
  scale_color_manual(values = c("blue", "green", "red"),
                     labels = c("Naive","SES","ETS")) +
  theme_bw() +
  geom_vline(xintercept=2001.8, linetype = "dashed", color = "black")
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#plot on test
st_models_test <- ts(data.frame(y1=window(st_prices, start=c(2001,t+1), frequency=365*24),
                                y2=naive_test,
                                y3=ses_test),
                                #y4=ets_test),
                     start=c(2001,t+1), frequency=365*24)
residuals_test <- ts(data.frame(y1=forecast_naive,
                                y2=forecast_ses),
                                #y3=forecast_ets),
                     start=c(2001,t+1), frequency=365*24)
datetest <- "2001/01/01"
p1 <- autoplot(st_models_test) +
  labs(x = "days", y = "prices", title = "Forecasts for DAM prices on test set") +
  scale_color_manual(values = c("black", "blue", "red"),#, "green"
                     labels = c("Ground truth", "Naive", "SES")) +#, "ETS"
  scale_x_continuous(breaks=seq(2001,2002,by=1/12), labels=seq(from = as.Date(datetest),
                                                             to = as.Date("2002/01/01"),
                                                             by = "month")) +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(residuals_test) +
  labs(x = "days", y = "residuals", title = paste("Residuals. MSE Naive = ", round(mse_naive_test,3),
                                                ", MSE SES = ", round(mse_ses,3)#,
                                                )) +#", MSE ETS = ", round(mse_ets_test,3)
  scale_color_manual(values = c("blue", "red"),#, "green"
                     labels = c("Naive             ","SES")) +#,"ETS"
  scale_x_continuous(breaks=seq(2001,2002,by=1/12), labels=seq(from = as.Date(datetest),
                                                               to = as.Date("2002/01/01"),
                                                               by = "month")) +
  theme_bw()
gridExtra::grid.arrange(p1, p2, ncol = 1)

#focus plot
h2 <- 8661
datefocus <- "2001/01/01"
st_models_focus <- ts(data.frame(y1=window(st_prices, start=c(2001,h2+1), frequency=365*24),
                                 y2=window(naive_test, start=c(2001,h2+1), frequency=365*24),
                                 y3=window(ses_test, start=c(2001,h2+1), frequency=365*24)),
                                 #y4=window(ets_test, start=c(2001,h2+1), frequency=365*24)),
                      start=c(2001,h2+1), frequency=365*24)
residuals_focus <- ts(data.frame(y1=window(forecast_naive, start=c(2001,h2+1), frequency=365*24),
                                 y2=window(forecast_ses, start=c(2001,h2+1), frequency=365*24)),
                                 #y3=window(forecast_ets, start=c(2001,h2+1), frequency=365*24)),
                      start=c(2001,h2+1), frequency=365*24)

p1 <- autoplot(st_models_focus) +
  labs(x = "days", y = "prices", title = paste("Forecasts for DAM prices on last ", length(st_volumes)-h2 ," hours")) +
  scale_color_manual(values = c("black", "blue", "red"),#, "cyan2"
                     labels = c("Ground truth", "Naive", "SES")) +#, "ETS"
  scale_x_continuous(breaks=seq(2001,2002,by=1/365), labels=seq(from = as.Date(datefocus),
                                                               to = as.Date("2002/01/01"),
                                                               by = "day")) +
  theme_bw()

#ses plot residuals
p2 <- autoplot(residuals_focus) +
  labs(x = "days", y = "residuals", title = paste("Residuals. MSE Naive = ", round(mse_naive_test,3),
                                                ", MSE SES = ", round(mse_ses,3))) +#,
                                                #", MSE ETS = ", round(mse_ets_test,3))) +
  scale_color_manual(values = c("blue", "red"),#, "cyan2"
                     labels = c("Naive            ","SES","ETS")) +
  scale_x_continuous(breaks=seq(2001,2002,by=1/365), labels=seq(from = as.Date(datefocus),
                                                                to = as.Date("2002/01/01"),
                                                                by = "day")) +
  theme_bw()
gridExtra::grid.arrange(p1, p2, ncol = 1)

#ACF
p1 <- ggAcf(st_prices, lag.max=100) +
      labs(x = 'lag', title = 'ACF: DAM prices') +
      theme_bw()
#PACF
p2 <- ggPacf(st_prices, lag.max=100) +
      labs(x = 'lag', title = 'PACF: DAM prices') +
      theme_bw()
gridExtra::grid.arrange(p1, p2, ncol = 1)

#
# ARIMA
#
arima_fit <- auto.arima(window(st_prices ,end=c(2001,t), frequency=365*24), seasonal = TRUE)#, seasonal = FALSE)
arima_fit

###
# ARIMA FORCED TO AR
###
p<-3
endd<-160
model <- Arima(window(st_prices ,end=c(2001,t), frequency=365*24), order=c(p,0,0))
fitted <- rep(0,length(st_prices))
for (i in 4:length(st_prices)){
  fitted[i] <- as.numeric(model$coef[3])*st_prices[i-3]+
               as.numeric(model$coef[2])*st_prices[i-2]+
               as.numeric(model$coef[1])*st_prices[i-1]+
               as.numeric(model$coef[4])*(1-as.numeric(model$coef[1])-
                                          as.numeric(model$coef[2])-
                                          as.numeric(model$coef[3]))
}
res_fitted = st_prices[4:length(st_prices)]-fitted[4:length(st_prices)]
fitted <- fitted[4:t]
fitted <- ts(fitted, start=c(2001,4), frequency=365*24)
forecast <- ts(forecast, start=c(2001,t+1), frequency=365*24)
mse_arima_train <- mean(res_fitted[1:(t-3)]^2)
mse_arima_test <- mean(res_fitted[(t-2):(length(st_prices)-3)]^2)
arima_model <- ts(c(as.vector(fitted), as.vector(forecast)),
                  start=c(2001,4), end=c(2002,1), frequency=365*24)
res_fitted <- ts(res_fitted, start=c(2001,4), end=c(2001,endd), frequency=365*24)
models_arima <- ts(data.frame(y1=window(st_prices, start=c(2001,4), frequency=365*24),
                              y2=arima_model), start=c(2001,4), end=c(2001,endd), frequency=365*24)

p1<-autoplot(models_arima) +
  labs(title = paste("AR(",p,")", sep = ""), x = "time (hours)", y = "prices", color = "") +
  scale_color_manual(values = c("black", "red"),
                     labels = c("Ground truth", paste("AR(",p,")", sep = ""))) +
  scale_x_continuous(breaks=seq(2001,2002,by=1/(12*30)), labels=seq(from = as.Date("2001/01/01"),
                                                                    to = as.Date("2001/12/27"),
                                                                    by = "day")) +
  theme_bw()
#ses plot residuals
p2 <- autoplot(res_fitted) +
  labs(title='Residuals', x = "days", y = "residuals", color = "") +
  scale_color_manual(values = c('AR(3)            ' = "red")) +
  geom_line(aes(colour = 'AR(3)            ')) +  # Aggiungo una mappatura del colore
  scale_x_continuous(breaks=seq(2001,2002,by=1/(12*30)), labels=seq(from = as.Date("2001/01/01"),
                                                                    to = as.Date("2001/12/27"),
                                                                    by = "day")) +
  theme_bw()
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)



###
# ARIMA FORCED TO AR APPLIED TO DAILY SERIES
###
p<-3
endd<-100
hour0 <- ts(matrix_prices[1,362:dim(matrix_prices)[2]], start=c(2000,362), frequency=365)
model <- Arima(hour0, order=c(p,0,0))
fitted <- rep(0,length(hour0))
for (i in 4:length(hour0)){
  fitted[i] <- as.numeric(model$coef[3])*hour0[i-3]+
    as.numeric(model$coef[2])*hour0[i-2]+
    as.numeric(model$coef[1])*hour0[i-1]+
    as.numeric(model$coef[4])*(1-as.numeric(model$coef[1])-
                                 as.numeric(model$coef[2])-
                                 as.numeric(model$coef[3]))
}
res_fitted = hour0[4:length(hour0)]-fitted[4:length(hour0)]
fitted <- fitted[4:length(hour0)]
fitted <- ts(fitted, start=c(2000,365), frequency=365)
mse_arima_train <- mean(res_fitted^2)
res_fitted <- ts(res_fitted, start=c(2000,365), end=c(2001,endd), frequency=365)
models_arima <- ts(data.frame(y1=window(hour0, start=c(2000,365), end=c(2001,endd), frequency=365),
                              y2=window(fitted, end=c(2001,endd), frequency=365)),
                   start=c(2000,365), end=c(2001,endd), frequency=365)

p1<-autoplot(models_arima) +
  labs(title = paste("AR(",p,") on DAM prices at hour 0", sep = ""), x = "time (days)", y = "prices", color = "") +
  scale_color_manual(values = c("black", "red"),
                     labels = c("Ground truth", paste("AR(",p,")", sep = ""))) +
  scale_x_continuous(breaks=seq(2001,2002,by=1/12), labels=seq(from = as.Date("2001/01/01"),
                                                                    to = as.Date("2002/01/01"),
                                                                    by = "month")) +
  theme_bw()
#ses plot residuals
p2 <- autoplot(res_fitted) +
  labs(title='Residuals', x = "days", y = "residuals", color = "") +
  scale_color_manual(values = c('AR(3)            ' = "red")) +
  geom_line(aes(colour = 'AR(3)            ')) +  # Aggiungo una mappatura del colore
  scale_x_continuous(breaks=seq(2001,2002,by=1/12), labels=seq(from = as.Date("2001/01/01"),
                                                               to = as.Date("2002/01/01"),
                                                               by = "month")) +
  theme_bw()
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#

h <- 2021
d1 <- 266
d2 <- 315
hour4 <- ts(matrix_prices[5,362:dim(matrix_prices)[2]], start=c(2000,362), frequency=365)
model <- Arima(hour4, order=c(p,0,0))
fitted <- rep(0,length(hour4))
for (i in 4:length(hour4)){
  fitted[i] <- as.numeric(model$coef[3])*hour4[i-3]+
    as.numeric(model$coef[2])*hour0[i-2]+
    as.numeric(model$coef[1])*hour0[i-1]+
    as.numeric(model$coef[4])*(1-as.numeric(model$coef[1])-
                                 as.numeric(model$coef[2])-
                                 as.numeric(model$coef[3]))
}
res_fitted = hour4[4:length(hour4)]-fitted[4:length(hour4)]
fitted <- fitted[4:length(hour4)]
fitted <- ts(fitted, start=c(2000,365), frequency=365)
mse_arima_train <- mean(res_fitted^2)
abs_fitted <- abs(res_fitted)
mae_ar_test <- mean(abs_fitted[(t+1):(length(hour4)-3)])
res_fitted <- ts(res_fitted, start=c(2000,365), frequency=365)
models_arima <- ts(data.frame(y1=window(hour4, start=c(h,d1), end=c(h,d2), frequency=365),
                              y2=window(fitted, start=c(h,d1), end=c(h,d2), frequency=365)),
                   start=c(h,d1), end=c(h,d2), frequency=365)
res_fitted_focus <- window(res_fitted, start=c(h,d1), end=c(h,d2))

p1<-autoplot(models_arima) +
  labs(title = paste("AR(",p,") on DAM prices at hour 4", sep = ""), x = "time (days)", y = "prices", color = "") +
  scale_color_manual(values = c("black", "red"),
                     labels = c("Ground truth", paste("AR(",p,")", sep = ""))) +
  scale_x_continuous(breaks=seq(2021,2022,by=1/12), labels=seq(from = as.Date("2021/01/01"),
                                                               to = as.Date("2022/01/01"),
                                                               by = "month")) +
  theme_bw()
#ses plot residuals
p2 <- autoplot(res_fitted_focus) +
  labs(title='Residuals4', x = "days", y = "residuals", color = "") +
  scale_color_manual(values = c('AR(3)            ' = "red")) +
  geom_line(aes(colour = 'AR(3)            ')) +  # Aggiungo una mappatura del colore
  scale_x_continuous(breaks=seq(2021,2022,by=1/12), labels=seq(from = as.Date("2021/01/01"),
                                                               to = as.Date("2022/01/01"),
                                                               by = "month")) +
  theme_bw()
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

# MAE

h <- 2021
d1 <- 266
d2 <- 315
p <- 3
mse_arima_test = rep(0,24)
mae_ar_test = rep(0,24)
for (z in 1:24){
  hour <- ts(matrix_prices[z,90:dim(matrix_prices)[2]], start=c(2000,90), frequency=365)
  hour_train <- window(hour, end=c(2017,237))
  model <- Arima(hour_train, order=c(p,0,0))
  fitted1 <- rep(0,length(hour))
  for (i in 4:length(hour)){
    fitted1[i] <- as.numeric(model$coef[3])*hour[i-3]+
      as.numeric(model$coef[2])*hour[i-2]+
      as.numeric(model$coef[1])*hour[i-1]+
      as.numeric(model$coef[4])*(1-as.numeric(model$coef[1])-
                                   as.numeric(model$coef[2])-
                                   as.numeric(model$coef[3]))
  }
  fitted1 = fitted1[4:length(hour)]
  res_fitted1 = hour[4:length(hour)]-fitted1
  res_ts1 <- ts(res_fitted1, start=c(2000,93), frequency=365)
  mse_arima_test[z] <- mean(window(res_ts1, start=c(2017,238))^2)
  abs_fitted1 <- abs(res_ts1)
  mae_ar_test[z] <- mean(window(abs_fitted1, start=c(2017,238)))
}

garch_AR3 <- garch(res_fitted1, order = c(1,3))

df_errors <- data.frame(y1=mse_arima_test,
                        y2=mae_ar_test)
write.csv(df_errors, "VSCODE/PyG/DAM/ar_errors.csv", row.names = FALSE)

h <- 2021
d1 <- 266
d2 <- 315

hour <- ts(matrix_prices[z,362:dim(matrix_prices)[2]], start=c(2000,362), frequency=365)
model <- Arima(hour, order=c(p,0,0))
fitted1 <- rep(0,length(hour))
for (i in 4:length(hour)){
  fitted1[i] <- as.numeric(model$coef[3])*hour[i-3]+
    as.numeric(model$coef[2])*hour[i-2]+
    as.numeric(model$coef[1])*hour[i-1]+
    as.numeric(model$coef[4])*(1-as.numeric(model$coef[1])-
                                 as.numeric(model$coef[2])-
                                 as.numeric(model$coef[3]))
}
res_fitted1 = hour[4:length(hour)]-fitted1[4:length(hour)]
fitted1 <- fitted1[4:length(hour)]
fitted1 <- ts(fitted1, start=c(2000,365), frequency=365)
mse_arima_train1 <- mean(res_fitted1^2)
abs_fitted1 <- abs(res_fitted1)
mae_ar_test1 <- mean(abs_fitted1[(t+1):(length(hour)-3)])
res_fitted1 <- ts(res_fitted1, start=c(2000,365), frequency=365)
models_arima1 <- ts(data.frame(y1=window(hour, start=c(h,d1), end=c(h,d2), frequency=365),
                              y2=window(fitted1, start=c(h,d1), end=c(h,d2), frequency=365)),
                   start=c(h,d1), end=c(h,d2), frequency=365)
res_fitted_focus1 <- window(res_fitted1, start=c(h,d1), end=c(h,d2))

p1<-autoplot(models_arima1) +
  labs(title = paste("AR(",p,") on DAM prices at hour 16", sep = ""), x = "time (days)", y = "prices", color = "") +
  scale_color_manual(values = c("black", "red"),
                     labels = c("Ground truth", paste("AR(",p,")", sep = ""))) +
  scale_x_continuous(breaks=seq(2021,2022,by=1/12), labels=seq(from = as.Date("2021/01/01"),
                                                               to = as.Date("2022/01/01"),
                                                               by = "month")) +
  theme_bw()
#ses plot residuals
p2 <- autoplot(res_fitted_focus1) +
  labs(title='Residuals', x = "days", y = "residuals", color = "") +
  scale_color_manual(values = c('AR(3)            ' = "red")) +
  geom_line(aes(colour = 'AR(3)            ')) +  # Aggiungo una mappatura del colore
  scale_x_continuous(breaks=seq(2021,2022,by=1/12), labels=seq(from = as.Date("2021/01/01"),
                                                               to = as.Date("2022/01/01"),
                                                               by = "month")) +
  theme_bw()
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

df_compare <- data.frame(y1=window(fitted, start=c(h,d1), end=c(h,d2), frequency=365),
                         y2=res_fitted_focus,
                         y3=window(fitted1, start=c(h,d1), end=c(h,d2), frequency=365),
                         y4=res_fitted_focus1)
write.csv(df_compare, "VSCODE/PyG/DAM/ar_compare.csv", row.names = FALSE)


#ARIMA(0,1,3)
res_train_arima <- arima_fit$residuals
fitted <- rep(0,length(st_prices))
res_fitted <- st_prices*0
res_fitted[1:3] <- res_train_arima[1:3]
ma <- rep(0,length(st_prices))
for (i in 4:length(st_prices)){
  ma[i] <- as.numeric(arima_fit$coef[3])*res_fitted[i-3]+as.numeric(arima_fit$coef[2])*res_fitted[i-2]+as.numeric(arima_fit$coef[1])*res_fitted[i-1]
  fitted[i] <- ma[i] + st_prices[i-1]
  res_fitted[i] <- st_prices[i] - fitted[i]
}
res_fitted <- res_fitted[2:length(st_prices)]
fitted[1:3] <- st_prices[1:3]
forecast <- fitted[(t+1):length(st_prices)]
fitted <- fitted[2:t]
fitted <- ts(fitted, start=c(2001,2), frequency=365*24)
forecast <- ts(forecast, start=c(2001,t+1), frequency=365*24)
mse_arima_train <- mean(res_fitted[1:(t-1)]^2)
mse_arima_test <- mean(res_fitted[t:(length(st_prices)-1)]^2)
arima_model <- ts(c(as.vector(fitted), as.vector(forecast)),
                  start=c(2001,2), end=c(2002,1), frequency=365*24)
res_fitted <- ts(res_fitted, start=c(2001,2), frequency=365*24)

#ARIMA(5,1,2)
res_train_arima <- arima_fit$residuals
fitted <- rep(0,length(st_prices))
res_fitted <- st_prices*0
res_fitted[1:6] <- res_train_arima[1:6]
ma <- rep(0,length(st_prices))
ar <- rep(0,length(st_prices))
c <- as.numeric(arima_fit$coef[8])*(1-as.numeric(arima_fit$coef[1])-
                                      as.numeric(arima_fit$coef[2])-
                                      as.numeric(arima_fit$coef[3])-
                                      as.numeric(arima_fit$coef[4])-
                                      as.numeric(arima_fit$coef[5]))
for (i in 7:length(st_prices)){
  ar[i] <- as.numeric(arima_fit$coef[1])*(st_prices[i-1]-st_prices[i-2])+
           as.numeric(arima_fit$coef[2])*(st_prices[i-2]-st_prices[i-3])+
           as.numeric(arima_fit$coef[3])*(st_prices[i-3]-st_prices[i-4])+
           as.numeric(arima_fit$coef[4])*(st_prices[i-4]-st_prices[i-5])+
           as.numeric(arima_fit$coef[5])*(st_prices[i-5]-st_prices[i-6])
  ma[i] <- as.numeric(arima_fit$coef[7])*res_fitted[i-2]+
           as.numeric(arima_fit$coef[6])*res_fitted[i-1]
  fitted[i] <- st_prices[i-1] + ma[i] + ar[i] + c
  res_fitted[i] <- st_prices[i] - fitted[i]
}
res_fitted <- res_fitted[2:length(st_prices)]
fitted[1:6] <- st_prices[1:6]
forecast <- fitted[(t+1):length(st_prices)]
fitted <- fitted[2:t]
fitted <- ts(fitted, start=c(2001,2), frequency=365*24)
forecast <- ts(forecast, start=c(2001,t+1), frequency=365*24)
mse_arima_train <- mean(res_fitted[1:(t-1)]^2)
mse_arima_test <- mean(res_fitted[t:(length(st_prices)-1)]^2)
arima_model <- ts(c(as.vector(fitted), as.vector(forecast)),
                  start=c(2001,2), end=c(2002,1), frequency=365*24)
res_fitted <- ts(res_fitted, start=c(2001,2), frequency=365*24)




# plot
models_arima <- ts(data.frame(y1=window(st_prices, start=c(2001,2), frequency=365*24),
                              y2=arima_model), start=c(2001,2), end=c(2002,1), frequency=365*24)

p1<-autoplot(models_arima) +
  labs(title = paste("ARIMA(5,1,2) Method: MSE_train = ", round(mse_arima_train,6), ", MSE_test = ", round(mse_arima_test,6)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "ARIMA")) +
  geom_vline(xintercept=2001.8, linetype = "dashed", color = "black") +
  scale_x_continuous(breaks=seq(2001,2002,by=1)) +
  theme_bw()
#ses plot residuals
p2 <- autoplot(res_fitted) +
  labs(x = "years", y = "residuals", title = "Residuals for DAM prices") +
  scale_color_manual(values = c("ARIMA           " = "blue")) +
  geom_line(aes(color = "ARIMA           ")) +  # Aggiungo una mappatura del colore
  geom_vline(xintercept=2001.8, linetype = "dashed", color = "black") +
  theme_bw()
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#focus
h2 <- 8661
st_models_focus <- ts(data.frame(y1=window(st_prices, start=c(2001,h2+1), frequency=365*24),
                                 y2=window(naive_test, start=c(2001,h2+1), frequency=365*24),
                                 y3=window(ses_test, start=c(2001,h2+1), frequency=365*24),
                                 y4=window(ets_test, start=c(2001,h2+1), frequency=365*24),
                                 y5=window(forecast, start=c(2001,h2+1), frequency=365*24)),
                      start=c(2001,h2+1), frequency=365*24)
residuals_focus <- ts(data.frame(y1=window(forecast_naive, start=c(2001,h2+1), frequency=365*24),
                                 y2=window(forecast_ses, start=c(2001,h2+1), frequency=365*24),
                                 y3=window(forecast_ets, start=c(2001,h2+1), frequency=365*24),
                                 y4=window(res_fitted, start=c(2001,h2+1), frequency=365*24)),
                      start=c(2001,h2+1), frequency=365*24)

datafocus <- "2001/01/01"
p1 <- autoplot(st_models_focus) +
  labs(x = "years", y = "prices", title = paste("Forecasts for DAM prices on last ", length(st_volumes)-h2 ," hours")) +
  scale_color_manual(values = c("black", "blue", "cyan2", "red", "purple"),
                     labels = c("Ground truth", "Naive", "SES", "ETS", "ARIMA")) +
  scale_x_continuous(breaks=seq(2001,2002,by=1/365), labels=seq(from = as.Date(datefocus),
                                                                to = as.Date("2002/01/01"),
                                                                by = "day")) +
  theme_bw()
#ses plot residuals
p2 <- autoplot(residuals_focus) +
  labs(x = "years", y = "residuals", title = paste("Residuals. MSE Naive = ", round(mse_naive_test,3),
                                                ", MSE SES = ", round(mse_ses,3),
                                                ", MSE ETS = ", round(mse_ets_test,3),
                                                ",\n                  MSE ARIMA = ", round(mse_arima_test,3))) +
  scale_color_manual(values = c("blue", "cyan2", "red", "purple"),
                     labels = c("Naive","SES","ETS", "ARIMA           ")) +
  scale_x_continuous(breaks=seq(2001,2002,by=1/365), labels=seq(from = as.Date(datefocus),
                                                                to = as.Date("2002/01/01"),
                                                                by = "day")) +
  theme_bw()
gridExtra::grid.arrange(p1, p2, ncol = 1)



########

arima_fit_forced <- auto.arima(window(st_prices ,end=c(2001,t), frequency=365*24),
                               seasonal = FALSE,
                               stepwise=FALSE,
                               approximation=FALSE)
arima_fit_forced
dt <- 3
prev <- rep(0,h+dt)
for (i in dt:(h+dt-1)){
  dx <- st_prices[(i-dt+1):i]
  refit <- Arima(dx, model=arima_fit)
  prev[i+1] <- forecast(refit, h=1)$mean
}
prev <- prev[(dt+1):(h+dt)]
arima_test <- ts(prev, start=c(2001,t+1), frequency=365*24)
forecast_arima = window(st_prices, start=c(2001,t+1)) - arima_test
mse_arima_train <- mean(res_train_arima^2)
mse_arima_test <- mean(forecast_arima^2)
arima_model <- ts(c(as.vector(arima_train), as.vector(arima_test)),
     start=c(2001,2), end=c(2002,1), frequency=365*24)
res_arima <- ts(c(as.vector(res_train_arima), as.vector(forecast_arima)),
              start=c(2001,2), end=c(2002,1), frequency=365*24)
#ets plot
models_arima <- ts(data.frame(y1=window(st_prices, start=c(2001,2), frequency=365*24),
                            y2=arima_model), start=c(2001,2), end=c(2002,1), frequency=365*24)

p1<-autoplot(models_arima) +
  labs(title = paste("ARIMA(0,1,3) Method: MSE_train = ", round(mse_arima_train,6), ", MSE_test = ", round(mse_arima_test,6)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "ARIMA")) +
  geom_vline(xintercept=2001.8, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(res_arima) +
  labs(x = "years", y = "prices", title = "Residuals for DAM prices") +
  scale_color_manual(values = c("ARIMA               " = "blue")) +
  geom_line(aes(color = "ARIMA               ")) +  # Aggiungo una mappatura del colore
  geom_vline(xintercept=2002.5, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

  
#arima_season <- auto.arima(window(st_prices ,end=c(2001,t), frequency=365*24), seasonal=TRUE)
#arima_season


############

#
#VAR
#
library(vars)
#var sulle differenze
t <- 6424 #t dati delle differenze
var_matrix <- t(matrix_prices[c(5,9,13,19,24),])
diff_matrix <- var_matrix[1:(dim(matrix_prices)[2]-1),]-var_matrix[2:dim(matrix_prices)[2],]
lag_select <- VARselect(diff_matrix[1:t,1], lag.max=7,
                        type="const")
#lag_select
optimal_lag <- as.numeric(lag_select$selection["AIC(n)"])
optimal_lag
vet <- c(5,9,13,18,23)
st_var_data <- ts(diff_matrix[1:t,], start=c(2000,2), frequency=365)
var_model <- VAR(st_var_data, p = optimal_lag, type = "const")
summary(var_model)
p <- as.numeric(var_model$p)
K <- as.numeric(var_model$K)
n_par = K*(1+K*p)
coef <- data.frame(matrix(NA, nrow = p, ncol = K))

#Series.1
coef[1,1:K] <- var_model$varresult$Series.1$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.1$coefficients[((l-1)*K+1):(l*K)]
}
coef
var_fitted <- rep(NA,dim(matrix_prices)[2]-1)
sum <- NA
for (time in p:(dim(matrix_prices)[2]-2)){
  sum <- as.numeric(var_model$varresult$Series.1$coefficients[n_par/K])
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*diff_matrix[time-k+1,j]
    }
  }
  var_fitted[time+1] <- sum
}

var_fitted[1:p] <- diff_matrix[1:p,1]
res_fitted <- diff_matrix[,1] - var_fitted
st_var <- ts(var_fitted, start=c(2000,2), frequency=365)
forecast <- var_fitted[(t+1):(dim(matrix_prices)[2]-1)]
fitted <- var_fitted[1:t]
fitted <- ts(fitted, start=c(2000,2), frequency=365)
forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_train <- mean(res_fitted[1:t]^2)
mse_var_test <- mean(res_fitted[(t+1):(dim(matrix_prices)[2]-1)]^2)
res_fitted <- ts(res_fitted, start=c(2000,2), frequency=365)
#plot
models_var <- ts(data.frame(y1=ts(diff_matrix[,1], start=c(2000,2), frequency=365),
                            y2=st_var), start=c(2000,2), frequency=365)

st_matrix_prices <- ts(diff_matrix[,1], start=c(2000,2), frequency=365)
#plot
p1<-autoplot(models_var) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#plot residuals
p2 <- autoplot(res_fitted) +
  labs(x = "years", y = "residuals", title = "Residuals for DAM prices") +
  scale_color_manual(values = c("VAR               " = "blue")) +
  geom_line(aes(color = "VAR               ")) +  # Aggiungo una mappatura del colore
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  ylim(-900,400) +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)
#serie 2 differenze
coef[1,1:K] <- var_model$varresult$Series.2$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.2$coefficients[((l-1)*K+1):(l*K)]
}
coef
for (time in p:(dim(matrix_prices)[2]-2)){
  sum <- as.numeric(var_model$varresult$Series.2$coefficients[n_par/K])
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*diff_matrix[time-k+1,j]
    }
  }
  var_fitted[time+1] <- sum
}

var_fitted[1:p] <- diff_matrix[1:p,2]
res_fitted <- diff_matrix[,2] - var_fitted
st_var <- ts(var_fitted, start=c(2000,2), frequency=365)
forecast <- var_fitted[(t+1):(dim(matrix_prices)[2]-1)]
fitted <- var_fitted[1:t]
fitted <- ts(fitted, start=c(2000,2), frequency=365)
forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_train <- mean(res_fitted[1:t]^2)
mse_var_test <- mean(res_fitted[(t+1):(dim(matrix_prices)[2]-1)]^2)
res_fitted <- ts(res_fitted, start=c(2000,2), frequency=365)
#
models_var <- ts(data.frame(y1=ts(diff_matrix[,2], start=c(2000,2), frequency=365),
                            y2=st_var), start=c(2000,2), frequency=365)
st_matrix_prices <- ts(diff_matrix[,2], start=c(2000,2), frequency=365)
#plot
p1<-autoplot(models_var) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#plot residuals
p2 <- autoplot(res_fitted) +
  labs(x = "years", y = "prices", title = "Residuals for DAM prices") +
  scale_color_manual(values = c("VAR                 " = "blue")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)




####
#var sui prezzi
t <- 6424
lag_select <- VARselect(t(matrix_prices[c(5,9,13,19,24),1:t]), lag.max=10,
                        type="const")
#lag_select
optimal_lag <- as.numeric(lag_select$selection["AIC(n)"])
optimal_lag
vet <- c(5,9,13,18,23)
st_var_data <- ts(t(matrix_prices[vet,1:t]), start=c(2000,1), frequency=365)
var_model <- VAR(st_var_data, p = optimal_lag, type = "const")
summary(var_model)
p <- as.numeric(var_model$p)
K <- as.numeric(var_model$K)
n_par = K*(1+K*p)
coef <- data.frame(matrix(NA, nrow = p, ncol = K))

#Series.1
coef[1,1:K] <- var_model$varresult$Series.1$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.1$coefficients[((l-1)*K+1):(l*K)]
}
coef
var_fitted <- rep(NA,dim(matrix_prices)[2])
sum <- NA
for (time in 7:(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.1$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted[1:7] <- matrix_prices[vet[1],1:7]
res_fitted <- matrix_prices[vet[1],] - var_fitted
st_var <- ts(var_fitted, start=c(2000,1), frequency=365)
forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
fitted <- var_fitted[1:t]
fitted <- ts(fitted, start=c(2000,1), frequency=365)
forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_train <- mean(res_fitted[1:t]^2)
mse_var_test <- mean(res_fitted[(t+1):dim(matrix_prices)[2]]^2)
res_fitted <- ts(res_fitted, start=c(2000,1), frequency=365)
#
models_var <- ts(data.frame(y1=ts(matrix_prices[vet[1],], start=c(2000,1), frequency=365),
                            y2=st_var), start=c(2000,1), frequency=365)

st_matrix_prices <- ts(matrix_prices[vet[1],], start=c(2000,1), frequency=365)
#plot
p1<-autoplot(models_var) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(res_fitted) +
  labs(x = "years", y = "residuals", title = "Residuals for DAM prices") +
  scale_color_manual(values = c("VAR               " = "blue")) +
  geom_line(aes(color = "VAR               ")) +  # Aggiungo una mappatura del colore
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#focus annuale
h3 <- 2018
h4 <- 2019
models_var_focus <- ts(data.frame(y1=window(st_matrix_prices, start=c(h3,1), end=c(h4,1)),
                                  y2=window(st_var, start=c(h3,1), end=c(h4,1))), start=c(h3,1), frequency=365)

res_fitted_focus <- window(res_fitted, start=c(h3,1), end=c(h4,1))
pt1<-autoplot(models_var_focus) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  scale_x_continuous(breaks=seq(h3,h4,by=1)) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
#ses plot residuals
p2 <- autoplot(res_fitted_focus) +
  labs(x = "years", y = "residuals", title = "Residuals for DAM prices") +
  scale_color_manual(values = c("VAR               " = "blue")) +
  geom_line(aes(color = "VAR               ")) +  # Aggiungo una mappatura del colore
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  scale_x_continuous(breaks=seq(h3,h4,by=1)) +
  theme_bw()
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(pt1, p2, ncol = 1)
plot1 <- autoplot(models_var_focus) +
  labs(title = paste("4:00am, MSE_train =", round(mse_var_train,0), ", MSE_test =", round(mse_var_test,0)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue")) +
  scale_x_continuous(breaks=seq(h3,h4,by=1)) +
  ylim(0,1000) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw() +
  theme(legend.position = "none")

#focus giornaliero
h <- 2019
d1 <- 1
d2 <- 200
models_var_focus <- ts(data.frame(y1=window(st_matrix_prices, start=c(h,d1), end=c(h,d2)),
                                  y2=window(st_var, start=c(h,d1), end=c(h,d2))), start=c(h,d1), frequency=365)

res_fitted_focus <- window(res_fitted, start=c(h,d1), end=c(h,d2))
p1<-autoplot(models_var_focus) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(res_fitted_focus) +
  labs(x = "years", y = "prices", title = "Residuals for DAM prices") +
  scale_color_manual(values = c("VAR                 " = "blue")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#Series.2
coef[1,1:K] <- var_model$varresult$Series.2$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.2$coefficients[((l-1)*K+1):(l*K)]
}
coef
for (time in 7:(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.2$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted[1:7] <- matrix_prices[vet[2],1:7]
res_fitted <- matrix_prices[vet[2],] - var_fitted
st_var <- ts(var_fitted, start=c(2000,1), frequency=365)
forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
fitted <- var_fitted[1:t]
fitted <- ts(fitted, start=c(2000,1), frequency=365)
forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_train <- mean(res_fitted[1:t]^2)
mse_var_test <- mean(res_fitted[(t+1):dim(matrix_prices)[2]]^2)
res_fitted <- ts(res_fitted, start=c(2000,1), frequency=365)
#ets plot
models_var <- ts(data.frame(y1=ts(matrix_prices[vet[2],], start=c(2000,1), frequency=365),
                            y2=st_var), start=c(2000,1), frequency=365)
st_matrix_prices <- ts(matrix_prices[vet[2],], start=c(2000,1), frequency=365)
#plot
p1<-autoplot(models_var) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(res_fitted) +
  labs(x = "years", y = "prices", title = "Residuals for DAM prices") +
  scale_color_manual(values = c("VAR                 " = "blue")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#focus annuale
h3 <- 2018
h4 <- 2019
models_var_focus <- ts(data.frame(y1=window(st_matrix_prices, start=c(h3,1), end=c(h4,1)),
                                  y2=window(st_var, start=c(h3,1), end=c(h4,1))), start=c(h3,1), frequency=365)

res_fitted_focus <- window(res_fitted, start=c(h3,1), end=c(h4,1))
p11<-autoplot(models_var_focus) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  scale_x_continuous(breaks=seq(h3,h4,by=1)) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
#ses plot residuals
p2 <- autoplot(res_fitted_focus) +
  labs(x = "years", y = "residuals", title = "Residuals for DAM prices") +
  scale_color_manual(values = c("VAR               " = "blue")) +
  geom_line(aes(color = "VAR               ")) +  # Aggiungo una mappatura del colore
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  scale_x_continuous(breaks=seq(h3,h4,by=1)) +
  theme_bw()
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p11, p2, ncol = 1)
plot2 <- autoplot(models_var_focus) +
  labs(title = paste("8:00am, MSE_train =", round(mse_var_train,0), ", MSE_test =", round(mse_var_test,0)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue")) +
  scale_x_continuous(breaks=seq(h3,h4,by=1)) +
  ylim(0,1000) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw() +
  theme(legend.position = "none")

#focus giornaliero
h <- 2019
d1 <- 100
d2 <- 150
models_var_focus <- ts(data.frame(y1=window(st_matrix_prices, start=c(h,d1), end=c(h,d2)),
                                  y2=window(st_var, start=c(h,d1), end=c(h,d2))), start=c(h,d1), frequency=365)

res_fitted_focus <- window(res_fitted, start=c(h,d1), end=c(h,d2))
p1<-autoplot(models_var_focus) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  scale_x_continuous(breaks=seq(h3,h4,by=1)) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
#ses plot residuals
p2 <- autoplot(res_fitted_focus) +
  labs(x = "years", y = "residuals", title = "Residuals for DAM prices") +
  scale_color_manual(values = c("VAR                 " = "blue")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#Series.3
coef[1,1:K] <- var_model$varresult$Series.3$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.3$coefficients[((l-1)*K+1):(l*K)]
}
coef
for (time in 7:(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.3$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted[1:7] <- matrix_prices[vet[3],1:7]
res_fitted <- matrix_prices[vet[3],] - var_fitted
st_var <- ts(var_fitted, start=c(2000,1), frequency=365)
forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
fitted <- var_fitted[1:t]
fitted <- ts(fitted, start=c(2000,1), frequency=365)
forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_train <- mean(res_fitted[1:t]^2)
mse_var_test <- mean(res_fitted[(t+1):dim(matrix_prices)[2]]^2)
res_fitted <- ts(res_fitted, start=c(2000,1), frequency=365)
#ets plot
models_var <- ts(data.frame(y1=ts(matrix_prices[vet[3],], start=c(2000,1), frequency=365),
                            y2=st_var), start=c(2000,1), frequency=365)
st_matrix_prices <- ts(matrix_prices[vet[3],], start=c(2000,1), frequency=365)
#plot
p1<-autoplot(models_var) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(res_fitted) +
  labs(x = "years", y = "prices", title = "Residuals for DAM prices") +
  scale_color_manual(values = c("VAR                 " = "blue")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#focus annuale
h3 <- 2018
h4 <- 2019
models_var_focus <- ts(data.frame(y1=window(st_matrix_prices, start=c(h3,1), end=c(h4,1)),
                                  y2=window(st_var, start=c(h3,1), end=c(h4,1))), start=c(h3,1), frequency=365)

res_fitted_focus <- window(res_fitted, start=c(h3,1), end=c(h4,1))
p111<-autoplot(models_var_focus) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  scale_x_continuous(breaks=seq(h3,h4,by=1)) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
#ses plot residuals
p2 <- autoplot(res_fitted_focus) +
  labs(x = "years", y = "residuals", title = "Residuals for DAM prices") +
  scale_color_manual(values = c("VAR               " = "blue")) +
  geom_line(aes(color = "VAR               ")) +  # Aggiungo una mappatura del colore
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  scale_x_continuous(breaks=seq(h3,h4,by=1)) +
  theme_bw()
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p111, p2, ncol = 1)
plot3 <- autoplot(models_var_focus) +
  labs(title = paste("12:00pm, MSE_train =", round(mse_var_train,0), ", MSE_test =", round(mse_var_test,0)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue")) +
  scale_x_continuous(breaks=seq(h3,h4,by=1)) +
  ylim(0,1000) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw() +
  theme(legend.position = "none")


#focus giornaliero
h <- 2019
d1 <- 100
d2 <- 150
models_var_focus <- ts(data.frame(y1=window(st_matrix_prices, start=c(h,d1), end=c(h,d2)),
                                  y2=window(st_var, start=c(h,d1), end=c(h,d2))), start=c(h,d1), frequency=365)

res_fitted_focus <- window(res_fitted, start=c(h,d1), end=c(h,d2))
p1<-autoplot(models_var_focus) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(res_fitted_focus) +
  labs(x = "years", y = "prices", title = "Residuals for DAM prices") +
  scale_color_manual(values = c("VAR                 " = "blue")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)


#Series.4
coef[1,1:K] <- var_model$varresult$Series.4$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.4$coefficients[((l-1)*K+1):(l*K)]
}
coef
for (time in 7:(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.4$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted[1:7] <- matrix_prices[vet[4],1:7]
res_fitted <- matrix_prices[vet[4],] - var_fitted
st_var <- ts(var_fitted, start=c(2000,1), frequency=365)
forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
fitted <- var_fitted[1:t]
fitted <- ts(fitted, start=c(2000,1), frequency=365)
forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_train <- mean(res_fitted[1:t]^2)
mse_var_test <- mean(res_fitted[(t+1):dim(matrix_prices)[2]]^2)
res_fitted <- ts(res_fitted, start=c(2000,1), frequency=365)
#ets plot
models_var <- ts(data.frame(y1=ts(matrix_prices[vet[4],], start=c(2000,1), frequency=365),
                            y2=st_var), start=c(2000,1), frequency=365)
st_matrix_prices <- ts(matrix_prices[vet[4],], start=c(2000,1), frequency=365)
#plot
p1<-autoplot(models_var) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(res_fitted) +
  labs(x = "years", y = "prices", title = "Residuals for DAM prices") +
  scale_color_manual(values = c("VAR                 " = "blue")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)


#focus annuale
h3 <- 2018
h4 <- 2019
models_var_focus <- ts(data.frame(y1=window(st_matrix_prices, start=c(h3,1), end=c(h4,1)),
                                  y2=window(st_var, start=c(h3,1), end=c(h4,1))), start=c(h3,1), frequency=365)

res_fitted_focus <- window(res_fitted, start=c(h3,1), end=c(h4,1))
p1111<-autoplot(models_var_focus) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  scale_x_continuous(breaks=seq(h3,h4,by=1)) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
#ses plot residuals
p2 <- autoplot(res_fitted_focus) +
  labs(x = "years", y = "residuals", title = "Residuals for DAM prices") +
  scale_color_manual(values = c("VAR               " = "blue")) +
  geom_line(aes(color = "VAR               ")) +  # Aggiungo una mappatura del colore
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  scale_x_continuous(breaks=seq(h3,h4,by=1)) +
  theme_bw()
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1111, p2, ncol = 1)
plot4 <- autoplot(models_var_focus) +
  labs(title = paste("6:00pm, MSE_train =", round(mse_var_train,0), ", MSE_test =", round(mse_var_test,0)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue")) +
  scale_x_continuous(breaks=seq(h3,h4,by=1)) +
  ylim(0,1000) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw() +
  theme(legend.position = "none")

#focus giornaliero
h <- 2019
d1 <- 200
d2 <- 250
models_var_focus <- ts(data.frame(y1=window(st_matrix_prices, start=c(h,d1), end=c(h,d2)),
                                  y2=window(st_var, start=c(h,d1), end=c(h,d2))), start=c(h,d1), frequency=365)

res_fitted_focus <- window(res_fitted, start=c(h,d1), end=c(h,d2))
p1<-autoplot(models_var_focus) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(res_fitted_focus) +
  labs(x = "years", y = "prices", title = "Residuals for DAM prices") +
  scale_color_manual(values = c("VAR                 " = "blue")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#Series.5
coef[1,1:K] <- var_model$varresult$Series.5$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.5$coefficients[((l-1)*K+1):(l*K)]
}
coef
for (time in 7:(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.5$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted[1:7] <- matrix_prices[vet[5],1:7]
res_fitted <- matrix_prices[vet[5],] - var_fitted
st_var <- ts(var_fitted, start=c(2000,1), frequency=365)
forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
fitted <- var_fitted[1:t]
fitted <- ts(fitted, start=c(2000,1), frequency=365)
forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_train <- mean(res_fitted[1:t]^2)
mse_var_test <- mean(res_fitted[(t+1):dim(matrix_prices)[2]]^2)
res_fitted <- ts(res_fitted, start=c(2000,1), frequency=365)
#ets plot
models_var <- ts(data.frame(y1=ts(matrix_prices[vet[5],], start=c(2000,1), frequency=365),
                            y2=st_var), start=c(2000,1), frequency=365)
st_matrix_prices <- ts(matrix_prices[vet[5],], start=c(2000,1), frequency=365)
#plot
p1<-autoplot(models_var) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(res_fitted) +
  labs(x = "years", y = "prices", title = "Residuals for DAM prices") +
  scale_color_manual(values = c("VAR                 " = "blue")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#focus annuale
h3 <- 2018
h4 <- 2019
models_var_focus <- ts(data.frame(y1=window(st_matrix_prices, start=c(h3,1), end=c(h4,1)),
                                  y2=window(st_var, start=c(h3,1), end=c(h4,1))), start=c(h3,1), frequency=365)

res_fitted_focus <- window(res_fitted, start=c(h3,1), end=c(h4,1))
p11111<-autoplot(models_var_focus) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  scale_x_continuous(breaks=seq(h3,h4,by=1)) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
#ses plot residuals
p2 <- autoplot(res_fitted_focus) +
  labs(x = "years", y = "residuals", title = "Residuals for DAM prices") +
  scale_color_manual(values = c("VAR               " = "blue")) +
  geom_line(aes(color = "VAR               ")) +  # Aggiungo una mappatura del colore
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  scale_x_continuous(breaks=seq(h3,h4,by=1)) +
  theme_bw()
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p11111, p2, ncol = 1)
plot5 <- autoplot(models_var_focus) +
  labs(title = paste("11:00pm, MSE_train =", round(mse_var_train,0), ", MSE_test =", round(mse_var_test,0)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth                                                                                                                 ", "VAR")) +
  scale_x_continuous(breaks=seq(h3,h4,by=1)) +
  ylim(0,1000) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()

#focus giornaliero
h <- 2019
d1 <- 200
d2 <- 250
models_var_focus <- ts(data.frame(y1=window(st_matrix_prices, start=c(h,d1), end=c(h,d2)),
                                  y2=window(st_var, start=c(h,d1), end=c(h,d2))), start=c(h,d1), frequency=365)

res_fitted_focus <- window(res_fitted, start=c(h,d1), end=c(h,d2))
p1<-autoplot(models_var_focus) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(res_fitted_focus) +
  labs(x = "years", y = "prices", title = "Residuals for DAM prices") +
  scale_color_manual(values = c("VAR                 " = "blue")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

pt1 <- gridExtra::grid.arrange(plot1, plot2, ncol=2)
pt2 <- gridExtra::grid.arrange(plot3, plot4, ncol=2)
ptt <- gridExtra::grid.arrange(pt1, pt2, plot5, ncol=1)


#volumes
diff_volumes <- st_volumes[2:length(st_volumes)]-st_volumes[1:(length(st_volumes)-1)]
diff_volumes <- ts(diff_volumes, start=c(2001,2), frequency=365*24)
#plot
p1 <- autoplot(st_volumes) +
  ggtitle("DAM volumes") +
  xlab("date") + ylab("volumes") +
  theme_bw() +
  guides(colour=guide_legend(title="volume"))
p2 <- autoplot(diff_volumes) +
  ggtitle("DAM diff prices") +
  xlab("date") + ylab("diff volumes") +
  theme_bw() +
  ylim(-1500,1500) +
  guides(colour=guide_legend(title="diff volumes"))
gridExtra::grid.arrange(p1, p2, ncol = 1)

#focus
autoplot(window(st_volumes, end=c(2001,24*5))) +
  ggtitle("DAM volumes") +
  xlab("date") + ylab("volumes") +
  theme_bw() +
  scale_x_continuous(breaks=seq(2001,2002,by=1/(12*30)), labels=seq(from = as.Date("2001/01/01"),
                                                             to = as.Date("2001/12/27"),
                                                             by = "day")) +
  guides(colour=guide_legend(title="volume"))

#
# SES
#
#h <- 4381
#t <- 13140
h <- 1752 #20%
t <- 7009 #80% train
ses_model <- ses(window(st_volumes,end=c(2001,t), frequency=365*24), h=1)
ses_model
alpha <- as.numeric(ses_model$model$par[1])
round(accuracy(ses_model),3)
res_ses_train <- ses_model$residuals
res_ses_train <- res_ses_train[2:t]

#ses forecasting
ses_train <- fitted(ses_model)
ses_train <- ses_train[2:t]
ses_test <- ts(rep(0,h))
sum <- as.numeric(ses_model$model$par[2])*(1-alpha)^t
for (j in 1:t){
  sum <- sum + st_volumes[j]*alpha*(1-alpha)^(t-j)
}
ses_test[1] <- sum
for (i in 1:(h-1)){
  ses_test[i+1] <- ses_test[i]*(1-alpha) + alpha*st_volumes[t+i]
}
ses_test <- ts(ses_test,start=c(2001,t+1),frequency=365*24)
#calcolo gli errori di forecast ed mse
forecast_ses <- window(st_volumes,start=c(2001,t+1))-ses_test
mse_ses <- mean(forecast_ses^2)
#creo un unico vettore per poterlo plottare
ses_volumes<-ts(c(as.vector(ses_train), as.vector(ses_test)),
               start=c(2001,2), end=c(2002,1), frequency=365*24)
res_ses <- ts(c(as.vector(res_ses_train), as.vector(forecast_ses)),
              start=c(2001,2), end=c(2002,1), frequency=365*24)

#ses plot
models_ses <- ts(data.frame(y1=window(st_volumes,start=c(2001,2), frequency=365*24),
                            y2=ses_volumes), start=c(2001,2), end=c(2002,1), frequency=365*24)
p1<-autoplot(models_ses) +
  labs(title = paste("Simple Exponential Smoothing: MSE_train = ", round(ses_model$model$mse,6), ", MSE_test = ", round(mse_ses,6)),
       x = "years", y = "volumes") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "SES")) +
  geom_vline(xintercept=2001.8, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(res_ses) +
  labs(x = "years", y = "volumes", title = "Residuals for DAM volumes") +
  scale_color_manual(values = c("SES          " = "blue")) +
  geom_line(aes(color = "SES          ")) +  # Aggiungo una mappatura del colore
  geom_vline(xintercept=2001.8, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#
# naive
#
naive_train <- fitted(naive(window(st_volumes, end=c(2001,t), frequency=365*24)))
#naive forecasting
naive_test <- ts(rep(0,h))
for (i in 1:h){
  naive_test[i] <- st_volumes[t+i-1]
}
#naive: forecast error, mse
naive_train <- naive_train[2:t]
naive_train <- ts(naive_train, start=c(2001,2), frequency=365*24)
naive_test <- ts(naive_test,start=c(2001,t+1),frequency=365*24)
forecast_naive <- window(st_volumes,start=c(2001,t+1), frequency=365*24) - naive_test
mse_naive_test <- mean(forecast_naive^2)
#naive: residual
res_naive_train <- window(window(st_volumes, start=c(2001,2), end=c(2001,t), frequency=365*24)-naive_train,
                          start=c(2001,2), frequency=365*24)
#naive: fitted accuracy
mse_naive_train <- mean(res_naive_train^2)
#creo un unico vettore per poterlo plottare
st_naive<-ts(c(as.vector(naive_train), as.vector(naive_test)),
             start=c(2001,2), end=c(2002,1), frequency=365*24)
res_naive <- ts(c(as.vector(res_naive_train), as.vector(forecast_naive)),
                start=c(2001,2), end=c(2002,1), frequency=365*24)
#naive plot
models_naive <- ts(data.frame(y1=window(st_volumes, start=c(2001,2), frequency=365*24),
                              y2=st_naive), start=c(2001,2), end=c(2002,1), frequency=365*24)

p1<-autoplot(models_naive) +
  labs(title = paste("Naive Method: MSE_train = ", round(mse_naive_train,6), ", MSE_test = ", round(mse_naive_test,6)),
       x = "years", y = "volumes") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "Naive")) +
  geom_vline(xintercept=2001.8, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(res_naive) +
  labs(x = "years", y = "volumes", title = "Residuals for DAM volumes") +
  scale_color_manual(values = c("Naive            " = "blue")) +
  geom_line(aes(color = "Naive            ")) +  # Aggiungo una mappatura del colore
  geom_vline(xintercept=2001.8, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#
# ETS
#
ets_model <- ets(window(st_volumes ,end=c(2001,t), frequency=365*24),
                 model="ZZZ", damped=NULL, alpha=NULL, beta=NULL,
                 gamma=NULL, phi=NULL, lambda=NULL, biasadj=FALSE,
                 additive.only=FALSE, restrict=TRUE,
                 allow.multiplicative.trend=FALSE)
summary(ets_model)
ets_train <- fitted(ets_model)
#ETS(M,Ad,N)
b <- ts(rep(0,length(st_volumes)))
l <- ts(rep(0,length(st_volumes)))
ets_train_fitted <- ts(rep(0,t-1))
ets_test <- ts(rep(0,h))
b[1] <- as.numeric(ets_model$par[5])
l[1] <- as.numeric(ets_model$par[4])
alpha <- as.numeric(ets_model$par[1])
beta <- as.numeric(ets_model$par[2])
phi <- as.numeric(ets_model$par[3])
for (i in 1:(t-1)){
  l[i+1] <- alpha*st_volumes[i]+(1-alpha)*(l[i]+phi*b[i])
  b[i+1] <- phi*b[i]*(1-beta)+beta*(l[i+1]-l[i])
  ets_train_fitted[i] <- l[i+1]+phi*b[i+1]
}
ets_train_fitted <- ts(ets_train_fitted, start=c(2001,2),end=c(2001,t), frequency=365*24)
for (j in 1:h){
  l[j+t] <- alpha*st_volumes[j+t-1]+(1-alpha)*(l[j+t-1]+phi*b[j+t-1])
  b[j+t] <- phi*b[j+t-1]*(1-beta)+beta*(l[j+t]-l[j+t-1])
  ets_test[j] <- l[j+t]+phi*b[j+t]
}
ets_test <- ts(ets_test, start=c(2001,t+1), frequency=365*24)
res_ets_train <- ts(st_volumes[2:t] - ets_train_fitted,
                    start=c(2001,2), end=c(2001,t), frequency=365*24)
forecast_ets <- ts(st_volumes[(t+1):length(st_volumes)] - ets_test,
                   start=c(2001,t+1), end=c(2002,1), frequency=365*24)

mse_ets_train <- mean(res_ets_train^2)
mse_ets_test <- mean(forecast_ets^2)

st_ets<-ts(c(as.vector(ets_train_fitted), as.vector(ets_test)),
           start=c(2001,2), end=c(2002,1), frequency=365*24)
res_ets <- ts(c(as.vector(res_ets_train), as.vector(forecast_ets)),
              start=c(2001,2), end=c(2002,1), frequency=365*24)
#ets plot
models_ets <- ts(data.frame(y1=window(st_volumes, start=c(2001,2), frequency=365*24),
                            y2=st_ets), start=c(2001,2), end=c(2002,1), frequency=365*24)

p1<-autoplot(models_ets) +
  labs(title = paste("ETS(M,Ad,N) Method: alpha = ",round(alpha,5),", beta = ",round(beta,5),", phi = ",round(phi,3)),
       x = "years", y = "volumes") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "ETS")) +
  geom_vline(xintercept=2001.8, linetype = "dashed", color = "black") +
  scale_x_continuous(breaks=seq(2001,2002,by=1)) +
  theme_bw()
#ses plot residuals
p2 <- autoplot(res_ets) +
  labs(x = "years", y = "residuals", title = paste("Residuals for DAM volumes: MSE_train = ", round(mse_ets_train,6), ", MSE_test = ", round(mse_ets_test,6))) +
  scale_color_manual(values = c("ETS               " = "blue")) +
  geom_line(aes(color = "ETS               ")) +  # Aggiungo una mappatura del colore
  geom_vline(xintercept=2001.8, linetype = "dashed", color = "black") +
  ylim(-1600,1700) +
  theme_bw()
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#compare plot models
st_models <- ts(data.frame(y1=window(st_volumes, start=c(2001,2), frequency=365*24),
                           y2=st_naive,
                           y3=ses_volumes,
                           y4=st_ets),
                start=c(2001,2), frequency=365*24)
residuals <- ts(data.frame(y1=res_naive,
                           y2=res_ses,
                           y3=res_ets),
                start=c(2001,2), frequency=365*24)
p1 <- autoplot(st_models) +
  labs(x = "years", y = "volumes", title = "Forecasts for DAM volumes") +
  scale_color_manual(values = c("black", "blue", "green", "red"),
                     labels = c("Ground truth", "Naive", "SES", "ETS")) +
  theme_bw() +
  geom_vline(xintercept=2001.8, linetype = "dashed", color = "black")
print(p1)
#ses plot residuals
p2 <- autoplot(residuals) +
  labs(x = "years", y = "residuals", title = "Residuals") +
  scale_color_manual(values = c("blue", "green", "red"),
                     labels = c("Naive","SES","ETS")) +
  theme_bw() +
  geom_vline(xintercept=2001.8, linetype = "dashed", color = "black")
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#plot on test
st_models_test <- ts(data.frame(y1=window(st_volumes, start=c(2001,t+1), frequency=365*24),
                                y2=naive_test,
                                y3=ses_test),
                                #y4=ets_test),
                     start=c(2001,t+1), frequency=365*24)
residuals_test <- ts(data.frame(y1=forecast_naive,
                                y2=forecast_ses),
                                #y3=forecast_ets),
                     start=c(2001,t+1), frequency=365*24)

p1 <- autoplot(st_models_test) +
  labs(x = "days", y = "volumes", title = "Forecasts for DAM volumes on test set") +
  scale_color_manual(values = c("black", "blue", "red"),#, "cyan2"
                     labels = c("Ground truth", "Naive", "SES")) +#, "ETS"
  scale_x_continuous(breaks=seq(2001,2002,by=1/12), labels=seq(from = as.Date(datetest),
                                                                to = as.Date("2002/01/01"),
                                                                by = "month")) +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(residuals_test) +
  labs(x = "days", y = "residuals", title = paste("Residuals. MSE Naive = ", round(mse_naive_test,3),
                                                ", MSE SES = ", round(mse_ses,3)#,
                                                )) +#", MSE ETS = ", round(mse_ets_test,3)
  scale_color_manual(values = c("blue", "red"),#, "cyan2"
                     labels = c("Naive            ","SES")) +#,"ETS"
  scale_x_continuous(breaks=seq(2001,2002,by=1/12), labels=seq(from = as.Date(datetest),
                                                               to = as.Date("2002/01/01"),
                                                               by = "month")) +
  ylim(-1250,1250) +
  theme_bw()
gridExtra::grid.arrange(p1, p2, ncol = 1)

#focus plot
h2 <- 8661
st_models_focus <- ts(data.frame(y1=window(st_volumes, start=c(2001,h2+1), frequency=365*24),
                                 y2=window(naive_test, start=c(2001,h2+1)),
                                 y3=window(ses_test, start=c(2001,h2+1)),
                                 y4=window(ets_test, start=c(2001,h2+1))),
                      start=c(2001,h2+1), frequency=365*24)
residuals_focus <- ts(data.frame(y1=window(forecast_naive, start=c(2001,h2+1)),
                                 y2=window(forecast_ses, start=c(2001,h2+1)),
                                 y3=window(forecast_ets, start=c(2001,h2+1))),
                      start=c(2001,h2+1), frequency=365*24)

p1 <- autoplot(st_models_focus) +
  labs(x = "years", y = "volumes", title = paste("Forecasts for DAM volumes on last ", length(st_volumes)-h2 ," hours")) +
  scale_color_manual(values = c("black", "blue", "green", "red"),
                     labels = c("Ground truth", "Naive", "SES", "ETS")) +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(residuals_focus) +
  labs(x = "years", y = "volumes", title = paste("Residuals. MSE Naive = ", round(mse_naive_test,3),
                                                ", MSE SES = ", round(mse_ses,3),
                                                ", MSE ETS = ", round(mse_ets_test,3))) +
  scale_color_manual(values = c("blue", "green", "red"),
                     labels = c("Naive           ","SES","ETS")) +
  theme_bw()
gridExtra::grid.arrange(p1, p2, ncol = 1)


#ACF
p1 <- ggAcf(st_volumes, lag.max=100) +
  labs(x = 'lag', title = 'ACF: DAM volumes') +
  theme_bw()
#PACF
p2 <- ggPacf(st_volumes, lag.max=100) +
  labs(x = 'lag', title = 'PACF: DAM volumes') +
  theme_bw()
gridExtra::grid.arrange(p1, p2, ncol = 1)
#ARIMA(2,0,0)

#
# ARIMA
#
arima_fit <- auto.arima(window(st_volumes ,end=c(2001,t), frequency=365*24), seasonal = TRUE)
arima_fit

#
# SARIMA
#
p <- 24
diff_seasonal <- diff(window(st_volumes ,end=c(2001,t), frequency=365*24),lag=p)
autoplot(window(diff_seasonal, end=c(2001,1000)))
ddiff_seasonal <- diff(diff_seasonal)
autoplot(window(ddiff_seasonal, end=c(2001,1000)))
#ACF_diff
p1 <- ggAcf(diff_seasonal, lag.max=100) +
  labs(x = 'lag', title = 'ACF: seasonal differenced DAM volumes') +
  theme_bw()
#PACF_diff
p2 <- ggPacf(diff_seasonal, lag.max=100) +
  labs(x = 'lag', title = 'PACF: seasonal differenced DAM volumes') +
  theme_bw()
gridExtra::grid.arrange(p1, p2, ncol = 1)
#da cui AR(2) nonseasonal e AR(2) seasonal, cioè ARIMA(2,0,0)(2,0,0)_24
#ACF_ddiff
p1 <- ggAcf(ddiff_seasonal, lag.max=100) +
  labs(x = 'lag', title = 'ACF: differenced seasonal differenced DAM volumes') +
  theme_bw()
#PACF_ddiff
p2 <- ggPacf(ddiff_seasonal, lag.max=100) +
  labs(x = 'lag', title = 'PACF: differenced seasonal differenced DAM volumes') +
  theme_bw()
gridExtra::grid.arrange(p1, p2, ncol = 1)
#da cui ARI(2) seasonal e ARI(1) nonseasonal / MA(1) o MA(2) nonseasonal e MA(2) seasonal 
x <- window(st_volumes ,end=c(2001,t), frequency=365*24)
model <- Arima(x, order=c(0,0,0), seasonal=list(order = c(2,1,2), period = p))
model <- Arima(x, order=c(0,1,0), seasonal=list(order = c(2,1,2), period = p))
model <- Arima(x, order=c(2,0,0), seasonal=list(order = c(2,1,2), period = p), method='CSS')
model <- Arima(x, order=c(2,1,0), seasonal=list(order = c(2,1,2), period = p), method='CSS')
model <- Arima(x, order=c(2,0,0), seasonal=list(order = c(2,1,0), period = p))
model

st_models_sarima <- ts(data.frame(y1=window(st_volumes, start=c(2001,t-150), end=c(2001,t), frequency=365*24),
                                  y2=model$fitted[(t-150):t]),
                       start=c(2001,t-150), frequency=365*24)
residuals_sarima <- ts(model$residuals[(t-150):t], start=c(2001,t-150), frequency=365*24)

p1 <- autoplot(st_models_sarima) +
  labs(x = "years", y = "volumes", title = "Forecasts for DAM volumes") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "Sarima")) +
  theme_bw()
#ses plot residuals
p2 <- autoplot(residuals_sarima) +
  labs(x = "years", y = "volumes", title = paste("Residuals. MSE = ", round(mean(model$residuals^2),3))) +
  scale_color_manual(values = c("blue"),
                     labels = c("Sarima")) +
  theme_bw()
gridExtra::grid.arrange(p1, p2, ncol = 1)

#ARIMA(2,0,0)(2,0,0)_24
start <- 3+p*3
ar1 <- as.numeric(model$coef[1])
ar2 <- as.numeric(model$coef[2])
sar1 <- as.numeric(model$coef[3])
sar2 <- as.numeric(model$coef[4])
c <- 0#as.numeric(model$coef[5])
residuals_sarima <- model$residuals
fitted <- rep(0,length(st_volumes))
for (i in start:length(st_volumes)){
  fitted[i] <- ar1*(st_volumes[i-1]-c) + ar2*(st_volumes[i-2]-c) + sar1*(st_volumes[i-p*1]-c) +
               sar2*(st_volumes[i-p*2]-c) - ar1*sar1*(st_volumes[i-p-1]-c) - ar2*sar1*(st_volumes[i-p-2]-c) -
               ar1*sar2*(st_volumes[i-2*p-1]-c) - ar2*sar2*(st_volumes[i-2*p-2]-c) + c
}

#ARIMA(2,0,0)(2,1,0)_24
start <- 3+p*3
ar1 <- as.numeric(model$coef[1])
ar2 <- as.numeric(model$coef[2])
sar1 <- as.numeric(model$coef[3])
sar2 <- as.numeric(model$coef[4])
sar1p <- 1+sar1 
sar12 <- sar1-sar2
residuals_sarima <- model$residuals
fitted <- rep(0,length(st_volumes))
for (i in start:length(st_volumes)){
  fitted[i] <- ar1*st_volumes[i-1] + ar2*st_volumes[i-2] + sar1p*st_volumes[i-p*1] -
    ar1*sar1p*st_volumes[i-p-1] - ar2*sar1p*st_volumes[i-p-2] - sar12*st_volumes[i-2*p] +
    ar1*sar12*st_volumes[i-2*p-1] + ar2*sar12*st_volumes[i-2*p-2] -
    sar2*st_volumes[i-3*p] + ar1*sar2*st_volumes[i-3*p-1] + ar2*sar2*st_volumes[i-3*p-2]
}

#ARIMA(2,0,0)(2,1,2)_24
start <- 3+p*3
ar1 <- as.numeric(model$coef[1])
ar2 <- as.numeric(model$coef[2])
sar1 <- as.numeric(model$coef[3])
sar2 <- as.numeric(model$coef[4])
sma1 <- as.numeric(model$coef[5])
sma2 <- as.numeric(model$coef[6])
sar1p <- 1+sar1
sar12 <- sar1-sar2
residuals_sarima <- st_volumes
residuals_sarima[1:t] <- model$residuals
fitted <- st_volumes
for (i in start:length(st_volumes)){
  fitted[i] <- ar1*st_volumes[i-1] + ar2*st_volumes[i-2] + sar1p*st_volumes[i-p*1] -
               ar1*sar1p*st_volumes[i-p-1] - ar2*sar1p*st_volumes[i-p-2] -
               sar12*st_volumes[i-2*p] + ar1*sar12*st_volumes[i-2*p-1] +
               ar2*sar12*st_volumes[i-2*p-2] - sar2*st_volumes[i-3*p] +
               ar1*sar2*st_volumes[i-3*p-1] + ar2*sar2*st_volumes[i-3*p-2] +
               sma1*residuals_sarima[i-p] + sma2*residuals_sarima[i-2*p]
  if (i>t){
    residuals_sarima[i] <- st_volumes[i]-fitted[i]
  }
}

#ARIMA(2,1,0)(2,1,2)_24
start <- 4+p*3
ar1 <- as.numeric(model$coef[1])
ar2 <- as.numeric(model$coef[2])
sar1 <- as.numeric(model$coef[3])
sar2 <- as.numeric(model$coef[4])
sma1 <- as.numeric(model$coef[5])
sma2 <- as.numeric(model$coef[6])
sar1p <- 1+sar1
ar1p <- 1+ar1
sar12 <- sar1-sar2
ar12 <- ar1-ar2
residuals_sarima <- st_volumes
residuals_sarima[1:t] <- model$residuals
fitted <- st_volumes
for (i in start:length(st_volumes)){
  fitted[i] <- ar1p*st_volumes[i-1] - ar12*st_volumes[i-2] - ar2*st_volumes[i-3] +
               sar1p*st_volumes[i-p] - ar1p*sar1p*st_volumes[i-p-1] + ar12*sar1p*st_volumes[i-p-2] +
               ar2*sar1p*st_volumes[i-p-3] - sar12*st_volumes[i-2*p] + ar1p*sar12*st_volumes[i-2*p-1] -
               ar12*sar12*st_volumes[i-2*p-2] - ar2*sar12*st_volumes[i-2*p-3] - sar2*st_volumes[i-3*p] +
               ar1p*sar2*st_volumes[i-3*p-1] - ar12*sar2*st_volumes[i-3*p-2] - ar2*sar2*st_volumes[i-3*p-3] +
               sma1*residuals_sarima[i-p] + sma2*residuals_sarima[i-2*p]
  if (i>t){
    residuals_sarima[i] <- st_volumes[i]-fitted[i]
  }
}


#ARIMA(0,1,0)(2,1,2)_24
start <- 2+p*3
sar1 <- as.numeric(model$coef[1])
sar2 <- as.numeric(model$coef[2])
sma1 <- as.numeric(model$coef[3])
sma2 <- as.numeric(model$coef[4])
sar1p <- 1+sar1 
sar12 <- sar1-sar2
residuals_sarima <- st_volumes
residuals_sarima[1:t] <- model$residuals
fitted <- rep(0,length(st_volumes))
for (i in start:length(st_volumes)){
  fitted[i] <- st_volumes[i-1] + sar1p*st_volumes[i-p] - sar1p*st_volumes[i-p-1] -
               sar12*st_volumes[i-2*p] + sar12*st_volumes[i-2*p-1] - sar2*st_volumes[i-3*p] +
               sar2*st_volumes[i-3*p-1] + sma1*residuals_sarima[i-p] + sma2*residuals_sarima[i-2*p]
  if (i>t){
    residuals_sarima[i] <- st_volumes[i]-fitted[i]
  }
}

#ARIMA(0,0,0)(2,1,2)_24
start <- 1+p*3
sar1 <- as.numeric(model$coef[1])
sar2 <- as.numeric(model$coef[2])
sma1 <- as.numeric(model$coef[3])
sma2 <- as.numeric(model$coef[4])
sar1p <- 1+sar1
sar12 <- sar1-sar2
residuals_sarima <- st_volumes
residuals_sarima[1:t] <- model$residuals
fitted <- st_volumes
for (i in start:length(st_volumes)){
  fitted[i] <- sar1p*st_volumes[i-p] - sar12*st_volumes[i-2*p] - sar2*st_volumes[i-3*p] +
               sma1*residuals_sarima[i-p] + sma2*residuals_sarima[i-2*p]
  if (i>t){
    residuals_sarima[i] <- st_volumes[i]-fitted[i]
  }
}


#ARIMA(0,1,2)(0,1,2)_24
start <- 3+p*3
ma1 <- as.numeric(model$coef[1])
ma2 <- as.numeric(model$coef[2])
sma1 <- as.numeric(model$coef[3])
sma2 <- as.numeric(model$coef[4])


#
fitted[1:start] <- st_volumes[1:start]
res_sarima_fitted <- st_volumes - fitted
res_sarima_fitted[1:start] <- residuals_sarima[1:start]
fitted <- fitted[2:length(st_volumes)]
sarima_model <- ts(fitted, start=c(2001,2), end=c(2002,1), frequency=365*24)
forecast <- fitted[t:(length(st_volumes)-1)]
fitted <- fitted[1:(t-1)]
fitted <- ts(fitted, start=c(2001,2), frequency=365*24)
forecast <- ts(forecast, start=c(2001,t+1), frequency=365*24)
mse_sarima_train <- mean(res_sarima_fitted[1:t]^2)
mse_sarima_test <- mean(res_sarima_fitted[(t+1):(length(st_volumes)-1)]^2)
res_sarima_fitted <- ts(res_sarima_fitted[2:length(st_volumes)], start=c(2001,2), frequency=365*24)

arima010212 <- sarima_model
res_arima010212 <- res_sarima_fitted

#plot
models_sarima <- ts(data.frame(y1=window(st_volumes, start=c(2001,2), frequency=365*24),
                              y2=sarima_model), start=c(2001,2), end=c(2002,1), frequency=365*24)

p1<-autoplot(models_sarima) +
  labs(title = paste("SARIMA(2,0,0)(2,1,0)_24 Method: MSE_train = ", round(mse_sarima_train,6), ", MSE_test = ", round(mse_sarima_test,6)),
       x = "years", y = "volumes") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "SARIMA")) +
  geom_vline(xintercept=2001.8, linetype = "dashed", color = "black") +
  scale_x_continuous(breaks=seq(2001,2002,by=1)) +
  theme_bw()
#plot residuals
p2 <- autoplot(res_sarima_fitted) +
  labs(x = "years", y = "residuals", title = "Residuals for DAM volumes") +
  scale_color_manual(values = c("SARIMA         " = "blue")) +
  geom_line(aes(color = "SARIMA         ")) +  # Aggiungo una mappatura del colore
  geom_vline(xintercept=2001.8, linetype = "dashed", color = "black") +
  ylim(-1500,1500) +
  theme_bw()
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#naive seasonal
naive_seasonal <- st_volumes
naive_seasonal[25:length(st_volumes)] <- st_volumes[1:(length(st_volumes)-24)]
naive_seasonal <- ts(naive_seasonal[2:length(st_volumes)], start=c(2001,2),frequency=365*24)
forecast_naive_seasonal <- st_volumes[2:length(st_volumes)] - naive_seasonal
forecast_naive_seasonal <- ts(forecast_naive_seasonal, start=c(2001,2),frequency=365*24)
mse_naive_seasonal <- mean(forecast_naive_seasonal[(t+1):length(forecast_naive_seasonal)]^2)

#ARIMA(2,1,0)
residuals_arima <- arima_fit$residuals
fitted <- rep(0,length(st_volumes))
ar <- rep(0,length(st_volumes))
c <- 0#as.numeric(arima_fit$coef[3])*(1-as.numeric(arima_fit$coef[2])-as.numeric(arima_fit$coef[1]))
for (i in 4:length(st_volumes)){
  ar[i] <- c+as.numeric(arima_fit$coef[2])*(st_volumes[i-2]-st_volumes[i-3])+as.numeric(arima_fit$coef[1])*(st_volumes[i-1]-st_volumes[i-2])
  fitted[i] <- ar[i] + st_volumes[i-1]
}
fitted[1:3] <- st_volumes[1:3]
res_fitted <- st_volumes - fitted
res_fitted[1:3] <- residuals_arima[1:3]
fitted <- fitted[2:length(st_volumes)]
arima_model <- ts(fitted, start=c(2001,2), end=c(2002,1), frequency=365*24)
forecast <- fitted[t:(length(st_volumes)-1)]
fitted <- fitted[1:(t-1)]
fitted <- ts(fitted, start=c(2001,2), frequency=365*24)
forecast <- ts(forecast, start=c(2001,t+1), frequency=365*24)
mse_arima_train <- mean(res_fitted[1:t]^2)
mse_arima_test <- mean(res_fitted[(t+1):length(st_volumes)]^2)
res_fitted <- ts(res_fitted[2:length(st_volumes)], start=c(2001,2), frequency=365*24)

models_arima <- ts(data.frame(y1=window(st_volumes, start=c(2001,2), frequency=365*24),
                              y2=arima_model), start=c(2001,2), end=c(2002,1), frequency=365*24)

arima210 <- arima_model
res_arima210 <- res_fitted

#focus
h2 <- 8711
st_models_focus <- ts(data.frame(y1=window(st_volumes, start=c(2001,h2+1), frequency=365*24),
                                 y2=window(naive_seasonal, start=c(2001,h2+1)),
                                 y3=window(ses_test, start=c(2001,h2+1)),
                                 y4=window(ets_test, start=c(2001,h2+1)),
                                 y5=window(arima_model, start=c(2001,h2+1), frequency=365*24),
                                 y6=window(sarima_model, start=c(2001,h2+1), frequency=365*24)),
                      start=c(2001,h2+1), frequency=365*24)
residuals_focus <- ts(data.frame(y1=window(forecast_naive_seasonal, start=c(2001,h2+1)),
                                 y2=window(forecast_ses, start=c(2001,h2+1)),
                                 y3=window(forecast_ets, start=c(2001,h2+1)),
                                 y4=window(res_fitted, start=c(2001,h2+1), frequency=365*24),
                                 y5=window(res_sarima_fitted, start=c(2001,h2+1), frequency=365*24)),
                      start=c(2001,h2+1), frequency=365*24)

p1 <- autoplot(st_models_focus) +
  labs(x = "days", y = "volumes", title = paste("Forecasts for DAM volumes on last ", length(st_volumes)-h2 ," hours")) +
  scale_color_manual(values = c("black", "blue", "cyan2", "yellow", "purple", "red"),
                     labels = c("Ground truth", "Naive Seasonal", "SES", "ETS", "ARIMA","SARIMA")) +
  scale_x_continuous(breaks=seq(2001,2002,by=1/365), labels=seq(from = as.Date("2001/01/01"),
                                                                to = as.Date("2002/01/01"),
                                                                by = "day")) +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(residuals_focus) +
  labs(x = "days", y = "residuals", title = paste("Residuals. MSE Naive Seasonal=", round(mse_naive_seasonal,0),
                                                  ", MSE SES=", round(mse_ses,0),
                                                  ", MSE ETS=", round(mse_ets_test,0),
                                                  ", MSE ARIMA=", round(mse_arima_test,0),
                                                  ", MSE SARIMA=", round(mse_sarima_test,0))) +
  scale_color_manual(values = c("blue", "cyan2", "yellow", "purple", "red"),
                     labels = c("Naive Seasonal","SES","ETS", "ARIMA", "SARIMA")) +
  scale_x_continuous(breaks=seq(2001,2002,by=1/365), labels=seq(from = as.Date("2001/01/01"),
                                                                to = as.Date("2002/01/01"),
                                                                by = "day")) +
  theme_bw()
gridExtra::grid.arrange(p1, p2, ncol = 1)



# focus between seasonal models
mse_naive_seasonal <- mean(window(forecast_naive_seasonal, start=c(2001,t+1))^2)
mse_arima210 <- mean(window(res_arima210, start=c(2001,t+1))^2)
mse_arima000212 <- mean(window(res_arima000212, start=c(2001,t+1))^2)
mse_arima010212 <- mean(window(res_arima010212, start=c(2001,t+1))^2)
mse_arima200210 <- mean(window(res_arima200210, start=c(2001,t+1))^2)
mse_arima200212 <- mean(window(res_arima200212, start=c(2001,t+1))^2)
mse_arima210212 <- mean(window(res_arima210212, start=c(2001,t+1))^2)

#focus
h2 <- 8745
st_models_focus <- ts(data.frame(y1=window(st_volumes, start=c(2001,h2+1), frequency=365*24),
                                 y2=window(naive_seasonal, start=c(2001,h2+1)),
                                 y3=window(arima210, start=c(2001,h2+1)),
                                 y4=window(arima000212, start=c(2001,h2+1)),
                                 y5=window(arima010212, start=c(2001,h2+1), frequency=365*24),
                                 y6=window(arima200210, start=c(2001,h2+1), frequency=365*24),
                                 y7=window(arima200212, start=c(2001,h2+1), frequency=365*24),
                                 y8=window(arima210212, start=c(2001,h2+1), frequency=365*24)),
                      start=c(2001,h2+1), frequency=365*24)
residuals_focus <- ts(data.frame(y1=window(forecast_naive_seasonal, start=c(2001,h2+1)),
                                 y2=window(res_arima210, start=c(2001,h2+1)),
                                 y3=window(res_arima000212, start=c(2001,h2+1)),
                                 y4=window(res_arima010212, start=c(2001,h2+1), frequency=365*24),
                                 y5=window(res_arima200210, start=c(2001,h2+1), frequency=365*24),
                                 y6=window(res_arima200212, start=c(2001,h2+1), frequency=365*24),
                                 y7=window(res_arima210212, start=c(2001,h2+1), frequency=365*24)),
                      start=c(2001,h2+1), frequency=365*24)

p1 <- autoplot(st_models_focus) +
  labs(x = "days", y = "volumes", title = paste("Forecasts for DAM volumes on last ", length(st_volumes)-h2 ," hours")) +
  scale_color_manual(values = c("black", "blue", "cyan2", "yellow", "purple", "red", "grey", "green"),
                     labels = c("Ground truth","Naive Seasonal","ARIMA(2,1,0)","ARIMA(0,0,0)(2,1,2)",
                                "ARIMA(0,1,0)(2,1,2)","ARIMA(2,0,0)(2,1,0)","ARIMA(2,0,0)(2,1,2)",
                                "ARIMA(2,1,0)(2,1,2)")) +
  scale_x_continuous(breaks=seq(2001,2002,by=1/365), labels=seq(from = as.Date("2001/01/01"),
                                                                to = as.Date("2002/01/01"),
                                                                by = "day")) +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(residuals_focus) +
  labs(x = "days", y = "residuals", title = paste("Residuals. MSE Naive S.=", round(mse_naive_seasonal,0),
                                                  ",\nMSE ARIMA(2,1,0)=", round(mse_arima210,0),
                                                  ", MSE ARIMA(0,0,0)(2,1,2)=", round(mse_arima000212,0),
                                                  ", MSE ARIMA(0,1,0)(2,1,2)=", round(mse_arima010212,0),
                                                  ", MSE ARIMA(2,0,0)(2,1,0)=", round(mse_arima200210,0),
                                                  ", MSE ARIMA(2,0,0)(2,1,2)=", round(mse_arima200212,0),
                                                  ", MSE ARIMA(2,1,0)(2,1,2)=", round(mse_arima210212,0))) +
  scale_color_manual(values = c("blue", "cyan2", "yellow", "purple", "red", "grey", "green"),
                     labels = c("Naive Seasonal","ARIMA(2,1,0)","ARIMA(0,0,0)(2,1,2)",
                                "ARIMA(0,1,0)(2,1,2)","ARIMA(2,0,0)(2,1,0)","ARIMA(2,0,0)(2,1,2)",
                                "ARIMA(2,1,0)(2,1,2)")) +
  scale_x_continuous(breaks=seq(2001,2002,by=1/365), labels=seq(from = as.Date("2001/01/01"),
                                                                to = as.Date("2002/01/01"),
                                                                by = "day")) +
  theme_bw()
gridExtra::grid.arrange(p1, p2, ncol = 1)

#focus
h2 <- 8711
st_models_focus <- ts(data.frame(y1=window(st_volumes, start=c(2001,h2+1), frequency=365*24),
                                 y2=window(arima210, start=c(2001,h2+1)),
                                 y3=window(arima000212, start=c(2001,h2+1)),
                                 y4=window(arima010212, start=c(2001,h2+1), frequency=365*24)),
                      start=c(2001,h2+1), frequency=365*24)
residuals_focus <- ts(data.frame(y1=window(res_arima210, start=c(2001,h2+1)),
                                 y2=window(res_arima000212, start=c(2001,h2+1)),
                                 y3=window(res_arima010212, start=c(2001,h2+1), frequency=365*24)),
                      start=c(2001,h2+1), frequency=365*24)

p1 <- autoplot(st_models_focus) +
  labs(x = "days", y = "volumes", title = paste("Forecasts for DAM volumes on last ", length(st_volumes)-h2 ," hours")) +
  scale_color_manual(values = c("black", "blue", "cyan2","red"),
                     labels = c("Ground truth","ARIMA(2,1,0)","ARIMA(0,0,0)(2,1,2)","ARIMA(0,1,0)(2,1,2)")) +
  scale_x_continuous(breaks=seq(2001,2002,by=1/365), labels=seq(from = as.Date("2001/01/01"),
                                                                to = as.Date("2002/01/01"),
                                                                by = "day")) +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(residuals_focus) +
  labs(x = "days", y = "residuals", title = paste("Residuals. MSE ARIMA(0,0,0)(2,1,2)=", round(mse_arima000212,0),
                                                  ",MSE ARIMA(2,1,0)=", round(mse_arima210,0),
                                                  ", MSE ARIMA(0,1,0)(2,1,2)=", round(mse_arima010212,0))) +
  scale_color_manual(values = c("blue", "cyan2","red"),
                     labels = c("ARIMA(2,1,0)","ARIMA(0,0,0)(2,1,2)","ARIMA(0,1,0)(2,1,2)")) +
  scale_x_continuous(breaks=seq(2001,2002,by=1/365), labels=seq(from = as.Date("2001/01/01"),
                                                                to = as.Date("2002/01/01"),
                                                                by = "day")) +
  ylim(-750,800) +
  theme_bw()
gridExtra::grid.arrange(p1, p2, ncol = 1)

#focus
h1 <- 7992
h2 <- 8016
st_models_focus <- ts(data.frame(y1=window(st_volumes, start=c(2001,h1+1), end=c(2001,h2+1)),
                                 y2=window(naive_seasonal, start=c(2001,h1+1), end=c(2001,h2+1)),
                                 y3=window(arima210, start=c(2001,h1+1), end=c(2001,h2+1)),
                                 y4=window(arima010212, start=c(2001,h1+1), end=c(2001,h2+1)),
                                 y5=window(arima200210, start=c(2001,h1+1), end=c(2001,h2+1)),
                                 y6=window(arima200212, start=c(2001,h1+1), end=c(2001,h2+1)),
                                 y7=window(arima210212, start=c(2001,h1+1), end=c(2001,h2+1))),
                      start=c(2001,h2+1), frequency=365*24)
residuals_focus <- ts(data.frame(y1=window(forecast_naive_seasonal, start=c(2001,h1+1), end=c(2001,h2+1)),
                                 y2=window(res_arima210, start=c(2001,h1+1), end=c(2001,h2+1)),
                                 y3=window(res_arima010212, start=c(2001,h1+1), end=c(2001,h2+1)),
                                 y4=window(res_arima200210, start=c(2001,h1+1), end=c(2001,h2+1)),
                                 y5=window(res_arima200212, start=c(2001,h1+1), end=c(2001,h2+1)),
                                 y6=window(res_arima210212, start=c(2001,h1+1), end=c(2001,h2+1))),
                      start=c(2001,h2+1), frequency=365*24)

p1 <- autoplot(st_models_focus) +
  labs(x = "days", y = "volumes", title = paste("Forecasts for DAM volumes focus on 1 December 2021")) +
  scale_color_manual(values = c("black", "blue", "cyan2", "yellow", "purple", "red", "grey"),
                     labels = c("Ground truth","Naive Seasonal","ARIMA(2,1,0)",
                                "ARIMA(0,1,0)(2,1,2)","ARIMA(2,0,0)(2,1,0)","ARIMA(2,0,0)(2,1,2)",
                                "ARIMA(2,1,0)(2,1,2)")) +
  scale_x_continuous(breaks=seq(2001,2002,by=1/365), labels=seq(from = as.Date("2001/01/01"),
                                                                to = as.Date("2002/01/01"),
                                                                by = "day")) +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(residuals_focus) +
  labs(x = "days", y = "residuals", title = paste("Residuals. MSE Naive S.=", round(mse_naive_seasonal,0),
                                                  ", MSE ARIMA(2,1,0)=", round(mse_arima210,0),
                                                  ", MSE ARIMA(0,1,0)(2,1,2)=", round(mse_arima010212,0),
                                                  ",\nMSE ARIMA(2,0,0)(2,1,0)=", round(mse_arima200210,0),
                                                  ", MSE ARIMA(2,0,0)(2,1,2)=", round(mse_arima200212,0),
                                                  ", MSE ARIMA(2,1,0)(2,1,2)=", round(mse_arima210212,0))) +
  scale_color_manual(values = c("blue", "cyan2", "yellow", "purple", "red", "grey"),
                     labels = c("Naive Seasonal","ARIMA(2,1,0)",
                                "ARIMA(0,1,0)(2,1,2)","ARIMA(2,0,0)(2,1,0)","ARIMA(2,0,0)(2,1,2)",
                                "ARIMA(2,1,0)(2,1,2)")) +
  scale_x_continuous(breaks=seq(2001,2002,by=1/365), labels=seq(from = as.Date("2001/01/01"),
                                                                to = as.Date("2002/01/01"),
                                                                by = "day")) +
  theme_bw()
gridExtra::grid.arrange(p1, p2, ncol = 1)



#
# ARIMA FORCED
#

order = 24*7
a <- rep(NA,order)
a[1:24 %% 24 != 0] <- 0
#a <- c(a,NA)

y <- window(st_volumes, start=c(2001,1), end=c(2001,t), frequency=365*24)
fitARlagged <- Arima(y, order=c(order,0,0),fixed=a,include.mean = FALSE)
fitARlagged
res_AR <- window(st_volumes, start=c(2001,1), end=c(2001,t), frequency=365*24)-fitARlagged$fitted
mse_AR <- mean(res_AR^2)
mse_AR

pred <- rep(0,t+h-order)
for (i in (order+1):(t+h)){
  v <- st_volumes[(i-1):(i-order)]
  #v <- c(v,1)
  pred[i-order] <- sum(v*as.numeric(fitARlagged$coef))
}

st_ARlagged <- ts(data.frame(y1=window(st_volumes, start=c(2001,order+1), frequency=365*24),
                             y2=pred),
                  start=c(2001,order+1), frequency=365*24)
residuals_ARlagged <- ts(data.frame(y1=window(st_volumes, start=c(2001,order+1), frequency=365*24)-pred),
                         start=c(2001,t+1), frequency=365*24)

mse_ARlagged_test <- mean(window(residuals_ARlagged, end=c(2001,t-order), frequency=365*24)^2)
p1 <- autoplot(st_ARlagged) +
  labs(x = "years", y = "volumes", title = "Forecasts for DAM volumes") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "Arima")) +
  theme_bw()
#ses plot residuals
p2 <- autoplot(residuals_ARlagged) +
  labs(x = "years", y = "volumes", title = paste("Residuals. MSE =", round(mse_sarima,3))) +
  scale_color_manual(values = c("blue"),
                     labels = c("Arima")) +
  theme_bw()
gridExtra::grid.arrange(p1, p2, ncol = 1)


#fitARlagged <- Arima(y, order=c(1,0,0),include.mean = FALSE)




#ARIMA(2,1,0)
p1<-autoplot(models_arima) +
  labs(title = paste("ARIMA(2,1,0) Method: MSE_train = ", round(mse_arima_train,6), ", MSE_test = ", round(mse_arima_test,6)),
       x = "years", y = "volumes") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "ARIMA")) +
  geom_vline(xintercept=2001.8, linetype = "dashed", color = "black") +
  scale_x_continuous(breaks=seq(2001,2002,by=1)) +
  theme_bw()
#plot residuals
p2 <- autoplot(res_fitted) +
  labs(x = "years", y = "residuals", title = "Residuals for DAM volumes") +
  scale_color_manual(values = c("ARIMA           " = "blue")) +
  geom_line(aes(color = "ARIMA           ")) +  # Aggiungo una mappatura del colore
  geom_vline(xintercept=2001.8, linetype = "dashed", color = "black") +
  ylim(-1600,1500) +
  theme_bw()
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#focus
h2 <- 8711
st_models_focus <- ts(data.frame(y1=window(st_volumes, start=c(2001,h2+1), frequency=365*24),
                                 y2=window(naive_test, start=c(2001,h2+1)),
                                 y3=window(ses_test, start=c(2001,h2+1)),
                                 y4=window(ets_test, start=c(2001,h2+1)),
                                 y5=window(arima_model, start=c(2001,h2+1), frequency=365*24)),
                      start=c(2001,h2+1), frequency=365*24)
residuals_focus <- ts(data.frame(y1=window(forecast_naive, start=c(2001,h2+1)),
                                 y2=window(forecast_ses, start=c(2001,h2+1)),
                                 y3=window(forecast_ets, start=c(2001,h2+1)),
                                 y4=window(res_fitted, start=c(2001,h2+1), frequency=365*24)),
                      start=c(2001,h2+1), frequency=365*24)

p1 <- autoplot(st_models_focus) +
  labs(x = "days", y = "volumes", title = paste("Forecasts for DAM volumes on last ", length(st_volumes)-h2 ," hours")) +
  scale_color_manual(values = c("black", "blue", "cyan2", "red", "purple"),
                     labels = c("Ground truth", "Naive", "SES", "ETS", "ARIMA")) +
  scale_x_continuous(breaks=seq(2001,2002,by=1/365), labels=seq(from = as.Date("2001/01/01"),
                                                                to = as.Date("2002/01/01"),
                                                                by = "day")) +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(residuals_focus) +
  labs(x = "days", y = "residuals", title = paste("Residuals. MSE Naive=", round(mse_naive_test,3),
                                                ", MSE SES=", round(mse_ses,3),
                                                ", MSE ETS=", round(mse_ets_test,3),
                                                ",\n                  MSE ARIMA=", round(mse_arima_test,3))) +
  scale_color_manual(values = c("blue", "cyan2", "red", "purple"),
                     labels = c("Naive","SES","ETS", "ARIMA")) +
  scale_x_continuous(breaks=seq(2001,2002,by=1/365), labels=seq(from = as.Date("2001/01/01"),
                                                                to = as.Date("2002/01/01"),
                                                                by = "day")) +
  theme_bw()
gridExtra::grid.arrange(p1, p2, ncol = 1)


h3 <- 2000
dt <- 50
st_models_train_focus <- ts(data.frame(y1=window(st_volumes,start=c(2001,h3+1),end=c(2001,h3+dt)),
                                       y2=window(naive_train,start=c(2001,h3+1),end=c(2001,h3+dt)),
                                       y3=window(ses_volumes,start=c(2001,h3+1),end=c(2001,h3+dt)),
                                       y4=window(ets_train_fitted,start=c(2001,h3+1),end=c(2001,h3+dt)),
                                       y5=window(arima_model,start=c(2001,h3+1),end=c(2001,h3+dt))),
                            start=c(2001,h3+1), frequency=365*24)
residuals_train_focus <- ts(data.frame(y1=window(res_naive,start=c(2001,h3+1),end=c(2001,h3+dt)),
                                       y2=window(res_ses,start=c(2001,h3+1),end=c(2001,h3+dt)),
                                       y3=window(res_ets,start=c(2001,h3+1),end=c(2001,h3+dt)),
                                       y4=window(res_fitted,start=c(2001,h3+1),end=c(2001,h3+dt))),
                            start=c(2001,h3+1), frequency=365*24)

p1 <- autoplot(st_models_train_focus) +
  labs(x = "years", y = "volumes", title = paste("Forecasts for DAM volumes on ", dt ," hours in Training set")) +
  scale_color_manual(values = c("black", "blue", "cyan2", "red", "purple"),
                     labels = c("Ground truth", "Naive", "SES", "ETS", "ARIMA")) +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(residuals_train_focus) +
  labs(x = "years", y = "volumes", title = paste("Residuals. MSE Naive=", round(mse_naive_train,3),
                                                ", MSE SES=", round(ses_model$model$mse,3),
                                                ", MSE ETS=", round(mse_ets_train,3),
                                                ", MSE ARIMA=", round(mse_arima_train,3))) +
  scale_color_manual(values = c("blue", "cyan2", "red", "purple"),
                     labels = c("Naive","SES","ETS", "ARIMA")) +
  theme_bw()
gridExtra::grid.arrange(p1, p2, ncol = 1)

#
# VAR
#
t <- 6424
lag_select <- VARselect(t(matrix_volumes[c(5,9,13,19,24),1:t]), lag.max=7,
                     type="const")
#lag_select
optimal_lag <- as.numeric(lag_select$selection["AIC(n)"])
optimal_lag
vet <- c(5,9,13,19,24)
st_var_data <- ts(t(matrix_volumes[vet,1:t]), start=c(2001,1), frequency=365*24)
var_model <- VAR(st_var_data, p = optimal_lag, type = "const")
summary(var_model)
p <- as.numeric(var_model$p)
K <- as.numeric(var_model$K)
n_par = K*(1+K*p)
coef <- data.frame(matrix(NA, nrow = p, ncol = K))

#v_h4
coef[1,1:K] <- var_model$varresult$v_h4$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$v_h4$coefficients[((l-1)*K+1):(l*K)]
}
coef
var_fitted <- rep(NA,dim(matrix_volumes)[2])
sum <- NA
for (time in 7:(dim(matrix_volumes)[2]-1)){
  sum <- var_model$varresult$v_h4$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_volumes[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted[1:7] <- matrix_volumes[vet[1],1:7]
res_fitted <- matrix_volumes[vet[1],] - var_fitted
st_var <- ts(var_fitted, start=c(2000,1), frequency=365)
forecast <- var_fitted[(t+1):dim(matrix_volumes)[2]]
fitted <- var_fitted[1:t]
fitted <- ts(fitted, start=c(2000,1), frequency=365)
forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_train <- mean(res_fitted[1:t]^2)
mse_var_test <- mean(res_fitted[(t+1):dim(matrix_volumes)[2]]^2)
res_fitted <- ts(res_fitted, start=c(2000,1), frequency=365)

models_var <- ts(data.frame(y1=ts(matrix_volumes[vet[1],], start=c(2000,1), frequency=365),
                              y2=st_var), start=c(2000,1), frequency=365)

st_matrix_volumes <- ts(matrix_volumes[vet[1],], start=c(2000,1), frequency=365)
#plot
p1<-autoplot(models_var) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "volumes") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#plot residuals
p2 <- autoplot(res_fitted) +
  labs(x = "years", y = "volumes", title = "Residuals for DAM volumes") +
  scale_color_manual(values = c("VAR                 " = "blue")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#focus annuale
h3 <- 2018
h4 <- 2019
models_var_focus <- ts(data.frame(y1=window(st_matrix_volumes, start=c(h3,1), end=c(h4,1)),
                                  y2=window(st_var, start=c(h3,1), end=c(h4,1))), start=c(h3,1), frequency=365)

res_fitted_focus <- window(res_fitted, start=c(h3,1), end=c(h4,1))
pt1<-autoplot(models_var_focus) +
  labs(title = paste("4:00am, MSE_train =", round(mse_var_train,0), ", MSE_test =", round(mse_var_test,0)),
       x = "years", y = "volumes") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  scale_x_continuous(breaks=seq(h3,h4,by=1)) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw() +
  ylim(8000,11500) +
  theme(legend.position = "none")
#plot residuals
p2 <- autoplot(res_fitted_focus) +
  labs(x = "years", y = "residuals", title = "Residuals for DAM volumes") +
  scale_color_manual(values = c("VAR                 " = "blue")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw() +
  ylim(-800,1200)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#focus giornaliero
h <- 2019
d1 <- 1
d2 <- 50
models_var_focus <- ts(data.frame(y1=window(st_matrix_volumes, start=c(h,d1), end=c(h,d2)),
                                  y2=window(st_var, start=c(h,d1), end=c(h,d2))), start=c(h,d1), frequency=365)

res_fitted_focus <- window(res_fitted, start=c(h,d1), end=c(h,d2))
p1<-autoplot(models_var_focus) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "volumes") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#plot residuals
p2 <- autoplot(res_fitted_focus) +
  labs(x = "years", y = "volumes", title = "Residuals for DAM volumes") +
  scale_color_manual(values = c("VAR                 " = "blue")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#v_h8
coef[1,1:K] <- var_model$varresult$v_h8$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$v_h8$coefficients[((l-1)*K+1):(l*K)]
}
coef
for (time in 7:(dim(matrix_volumes)[2]-1)){
  sum <- var_model$varresult$v_h8$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_volumes[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted[1:7] <- matrix_volumes[vet[2],1:7]
res_fitted <- matrix_volumes[vet[2],] - var_fitted
st_var <- ts(var_fitted, start=c(2000,1), frequency=365)
forecast <- var_fitted[(t+1):dim(matrix_volumes)[2]]
fitted <- var_fitted[1:t]
fitted <- ts(fitted, start=c(2000,1), frequency=365)
forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_train <- mean(res_fitted[1:t]^2)
mse_var_test <- mean(res_fitted[(t+1):dim(matrix_volumes)[2]]^2)
res_fitted <- ts(res_fitted, start=c(2000,1), frequency=365)

models_var <- ts(data.frame(y1=ts(matrix_volumes[vet[2],], start=c(2000,1), frequency=365),
                            y2=st_var), start=c(2000,1), frequency=365)
st_matrix_volumes <- ts(matrix_volumes[vet[2],], start=c(2000,1), frequency=365)
#plot
p1<-autoplot(models_var) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "volumes") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#plot residuals
p2 <- autoplot(res_fitted) +
  labs(x = "years", y = "volumes", title = "Residuals for DAM volumes") +
  scale_color_manual(values = c("VAR                 " = "blue")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#focus annuale
h3 <- 2018
h4 <- 2019
models_var_focus <- ts(data.frame(y1=window(st_matrix_volumes, start=c(h3,1), end=c(h4,1)),
                                  y2=window(st_var, start=c(h3,1), end=c(h4,1))), start=c(h3,1), frequency=365)

res_fitted_focus <- window(res_fitted, start=c(h3,1), end=c(h4,1))
pt2<-autoplot(models_var_focus) +
  labs(title = paste("8:00am, MSE_train =", round(mse_var_train,0), ", MSE_test =", round(mse_var_test,0)),
       x = "years", y = "volumes") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  scale_x_continuous(breaks=seq(h3,h4,by=1)) +
  ylim(8000,11500) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw() +
  theme(legend.position = "none")
#plot residuals
p2 <- autoplot(res_fitted_focus) +
  labs(x = "years", y = "residuals", title = "Residuals for DAM volumes") +
  scale_color_manual(values = c("VAR                 " = "blue")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  ylim(-1500,1500) +
  theme_bw()
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(pt2, p2, ncol = 1)

#focus giornaliero
h <- 2019
d1 <- 1
d2 <- 50
models_var_focus <- ts(data.frame(y1=window(st_matrix_volumes, start=c(h,d1), end=c(h,d2)),
                                  y2=window(st_var, start=c(h,d1), end=c(h,d2))), start=c(h,d1), frequency=365)

res_fitted_focus <- window(res_fitted, start=c(h,d1), end=c(h,d2))
p1<-autoplot(models_var_focus) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "volumes") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#plot residuals
p2 <- autoplot(res_fitted_focus) +
  labs(x = "years", y = "volumes", title = "Residuals for DAM volumes") +
  scale_color_manual(values = c("VAR                 " = "blue")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#v_h12
coef[1,1:K] <- var_model$varresult$v_h12$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$v_h12$coefficients[((l-1)*K+1):(l*K)]
}
coef
for (time in 7:(dim(matrix_volumes)[2]-1)){
  sum <- var_model$varresult$v_h12$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_volumes[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted[1:7] <- matrix_volumes[vet[3],1:7]
res_fitted <- matrix_volumes[vet[3],] - var_fitted
st_var <- ts(var_fitted, start=c(2000,1), frequency=365)
forecast <- var_fitted[(t+1):dim(matrix_volumes)[2]]
fitted <- var_fitted[1:t]
fitted <- ts(fitted, start=c(2000,1), frequency=365)
forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_train <- mean(res_fitted[1:t]^2)
mse_var_test <- mean(res_fitted[(t+1):dim(matrix_volumes)[2]]^2)
res_fitted <- ts(res_fitted, start=c(2000,1), frequency=365)

models_var <- ts(data.frame(y1=ts(matrix_volumes[vet[3],], start=c(2000,1), frequency=365),
                            y2=st_var), start=c(2000,1), frequency=365)
st_matrix_volumes <- ts(matrix_volumes[vet[3],], start=c(2000,1), frequency=365)
#plot
p1<-autoplot(models_var) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "volumes") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#plot residuals
p2 <- autoplot(res_fitted) +
  labs(x = "years", y = "volumes", title = "Residuals for DAM volumes") +
  scale_color_manual(values = c("VAR                 " = "blue")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#focus annuale
h3 <- 2018
h4 <- 2019
models_var_focus <- ts(data.frame(y1=window(st_matrix_volumes, start=c(h3,1), end=c(h4,1)),
                                  y2=window(st_var, start=c(h3,1), end=c(h4,1))), start=c(h3,1), frequency=365)

res_fitted_focus <- window(res_fitted, start=c(h3,1), end=c(h4,1))
pt3<-autoplot(models_var_focus) +
  labs(title = paste("12:00pm, MSE_train =", round(mse_var_train,0), ", MSE_test =", round(mse_var_test,0)),
       x = "years", y = "volumes") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  scale_x_continuous(breaks=seq(h3,h4,by=1)) +
  ylim(8000,11500) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw() +
  theme(legend.position = "none")
#plot residuals
p2 <- autoplot(res_fitted_focus) +
  labs(x = "years", y = "residuals", title = "Residuals for DAM volumes") +
  scale_color_manual(values = c("VAR                 " = "blue")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  ylim(-1250,1250) +
  theme_bw()
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(pt3, p2, ncol = 1)

#focus giornaliero
h <- 2019
d1 <- 1
d2 <- 50
models_var_focus <- ts(data.frame(y1=window(st_matrix_volumes, start=c(h,d1), end=c(h,d2)),
                                  y2=window(st_var, start=c(h,d1), end=c(h,d2))), start=c(h,d1), frequency=365)

res_fitted_focus <- window(res_fitted, start=c(h,d1), end=c(h,d2))
p1<-autoplot(models_var_focus) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "volumes") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#plot residuals
p2 <- autoplot(res_fitted_focus) +
  labs(x = "years", y = "volumes", title = "Residuals for DAM volumes") +
  scale_color_manual(values = c("VAR                 " = "blue")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#v_h18
coef[1,1:K] <- var_model$varresult$v_h18$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$v_h18$coefficients[((l-1)*K+1):(l*K)]
}
coef
for (time in 7:(dim(matrix_volumes)[2]-1)){
  sum <- var_model$varresult$v_h18$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_volumes[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted[1:7] <- matrix_volumes[vet[4],1:7]
res_fitted <- matrix_volumes[vet[4],] - var_fitted
st_var <- ts(var_fitted, start=c(2000,1), frequency=365)
forecast <- var_fitted[(t+1):dim(matrix_volumes)[2]]
fitted <- var_fitted[1:t]
fitted <- ts(fitted, start=c(2000,1), frequency=365)
forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_train <- mean(res_fitted[1:t]^2)
mse_var_test <- mean(res_fitted[(t+1):dim(matrix_volumes)[2]]^2)
res_fitted <- ts(res_fitted, start=c(2000,1), frequency=365)

models_var <- ts(data.frame(y1=ts(matrix_volumes[vet[4],], start=c(2000,1), frequency=365),
                            y2=st_var), start=c(2000,1), frequency=365)
st_matrix_volumes <- ts(matrix_volumes[vet[4],], start=c(2000,1), frequency=365)
#plot
p1<-autoplot(models_var) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "volumes") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(res_fitted) +
  labs(x = "years", y = "volumes", title = "Residuals for DAM volumes") +
  scale_color_manual(values = c("VAR                 " = "blue")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#focus annuale
h3 <- 2018
h4 <- 2019
models_var_focus <- ts(data.frame(y1=window(st_matrix_volumes, start=c(h3,1), end=c(h4,1)),
                                  y2=window(st_var, start=c(h3,1), end=c(h4,1))), start=c(h3,1), frequency=365)

res_fitted_focus <- window(res_fitted, start=c(h3,1), end=c(h4,1))
pt4<-autoplot(models_var_focus) +
  labs(title = paste("6:00pm, MSE_train =", round(mse_var_train,0), ", MSE_test =", round(mse_var_test,0)),
       x = "years", y = "volumes") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  scale_x_continuous(breaks=seq(h3,h4,by=1)) +
  ylim(8000,11500) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw() +
  theme(legend.position = "none")
#plot residuals
p2 <- autoplot(res_fitted_focus) +
  labs(x = "years", y = "residuals", title = "Residuals for DAM volumes") +
  scale_color_manual(values = c("VAR                 " = "blue")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  ylim(-1500,1500) +
  theme_bw()
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(pt4, p2, ncol = 1)

#focus giornaliero
h <- 2019
d1 <- 1
d2 <- 50
models_var_focus <- ts(data.frame(y1=window(st_matrix_volumes, start=c(h,d1), end=c(h,d2)),
                                  y2=window(st_var, start=c(h,d1), end=c(h,d2))), start=c(h,d1), frequency=365)

res_fitted_focus <- window(res_fitted, start=c(h,d1), end=c(h,d2))
p1<-autoplot(models_var_focus) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "volumes") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(res_fitted_focus) +
  labs(x = "years", y = "volumes", title = "Residuals for DAM volumes") +
  scale_color_manual(values = c("VAR                 " = "blue")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#v_h23
coef[1,1:K] <- var_model$varresult$v_h23$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$v_h23$coefficients[((l-1)*K+1):(l*K)]
}
coef
for (time in 7:(dim(matrix_volumes)[2]-1)){
  sum <- var_model$varresult$v_h23$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_volumes[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted[1:7] <- matrix_volumes[vet[5],1:7]
res_fitted <- matrix_volumes[vet[5],] - var_fitted
st_var <- ts(var_fitted, start=c(2000,1), frequency=365)
forecast <- var_fitted[(t+1):dim(matrix_volumes)[2]]
fitted <- var_fitted[1:t]
fitted <- ts(fitted, start=c(2000,1), frequency=365)
forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_train <- mean(res_fitted[1:t]^2)
mse_var_test <- mean(res_fitted[(t+1):dim(matrix_volumes)[2]]^2)
res_fitted <- ts(res_fitted, start=c(2000,1), frequency=365)
#ets plot
models_var <- ts(data.frame(y1=ts(matrix_volumes[vet[5],], start=c(2000,1), frequency=365),
                            y2=st_var), start=c(2000,1), frequency=365)
st_matrix_volumes <- ts(matrix_volumes[vet[5],], start=c(2000,1), frequency=365)
#plot
p1<-autoplot(models_var) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "volumes") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(res_fitted) +
  labs(x = "years", y = "volumes", title = "Residuals for DAM volumes") +
  scale_color_manual(values = c("VAR                 " = "blue")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#focus annuale
h3 <- 2018
h4 <- 2019
models_var_focus <- ts(data.frame(y1=window(st_matrix_volumes, start=c(h3,1), end=c(h4,1)),
                                  y2=window(st_var, start=c(h3,1), end=c(h4,1))), start=c(h3,1), frequency=365)

res_fitted_focus <- window(res_fitted, start=c(h3,1), end=c(h4,1))
pt5<-autoplot(models_var_focus) +
  labs(title = paste("11:00pm, MSE_train =", round(mse_var_train,0), ", MSE_test =", round(mse_var_test,0)),
       x = "years", y = "volumes") +
  scale_x_continuous(breaks=seq(h3,h4,by=1)) +
  ylim(8000,11500) +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth                                                                                                                 ", "VAR")) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
#plot residuals
p2 <- autoplot(res_fitted_focus) +
  labs(x = "years", y = "residuals", title = "Residuals for DAM volumes") +
  scale_color_manual(values = c("VAR                 " = "blue")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  ylim(-1000,1000) +
  theme_bw()
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(pt5, p2, ncol = 1)

#focus giornaliero
h <- 2019
d1 <- 1
d2 <- 50
models_var_focus <- ts(data.frame(y1=window(st_matrix_volumes, start=c(h,d1), end=c(h,d2)),
                                  y2=window(st_var, start=c(h,d1), end=c(h,d2))), start=c(h,d1), frequency=365)

res_fitted_focus <- window(res_fitted, start=c(h,d1), end=c(h,d2))
p1<-autoplot(models_var_focus) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train =", round(mse_var_train,6), ", MSE_test =", round(mse_var_test,6)),
       x = "years", y = "volumes") +
  scale_color_manual(values = c("black", "blue"),
                     labels = c("Ground truth", "VAR")) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(res_fitted_focus) +
  labs(x = "years", y = "volumes", title = "Residuals for DAM volumes") +
  scale_color_manual(values = c("VAR                 " = "blue")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)


plot1 <- gridExtra::grid.arrange(pt1, pt2, ncol=2)
plot2 <- gridExtra::grid.arrange(pt3, pt4, ncol=2)
ptt <- gridExtra::grid.arrange(plot1, plot2, pt5, ncol=1)
