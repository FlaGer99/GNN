####
#var sui prezzi: tutto il vettore
library(vars)
library(fpp2)
t <- 6442
load("matrix_prices.RData")
lag_select <- VARselect(t(matrix_prices[,1:t]), lag.max=20,
                        type="const")
#lag_select
optimal_lag <- as.numeric(lag_select$selection["AIC(n)"])
optimal_lag <- 7
vet <- c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)
st_var_data <- ts(t(matrix_prices[vet,90:t]), start=c(2000,90), frequency=365)
var_model <- VAR(st_var_data, p = optimal_lag, type = "const")
summary(var_model)
p <- as.numeric(var_model$p)
K <- as.numeric(var_model$K)
n_par = K*(1+K*p)
coef <- data.frame(matrix(NA, nrow = p, ncol = K))

mae_var_test <- rep(0,24)
mse_var_test <- rep(0,24)

var_test <- data.frame(matrix(ncol = 0, nrow = 7934))

#Series.1
coef[1,1:K] <- var_model$varresult$Series.1$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.1$coefficients[((l-1)*K+1):(l*K)]
}
coef
var_fitted <- rep(NA,dim(matrix_prices)[2])
sum <- NA
for (time in (90+optimal_lag-1):(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.1$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted <- var_fitted[(90+optimal_lag):dim(matrix_prices)[2]]
res_fitted <- matrix_prices[vet[1],(90+optimal_lag):dim(matrix_prices)[2]] - var_fitted
st_var <- ts(var_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
res_ts <- ts(res_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
var_test$Hour0 <- st_var
#forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
#fitted <- var_fitted[1:t]
#fitted <- ts(fitted, start=c(2000,1), frequency=365)
#forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_test[1] <- mean(window(res_ts, start=c(2017,238))^2)
abs_fitted <- abs(window(res_ts, start=c(2017,238)))
mae_var_test[1] <- mean(abs_fitted)

models_var <- ts(data.frame(y1=ts(matrix_prices[vet[1],(90+optimal_lag):dim(matrix_prices)[2]], start=c(2000,(90+optimal_lag)), frequency=365),
                            y2=st_var), start=c(2000,(90+optimal_lag)), frequency=365)

st_matrix_prices <- ts(matrix_prices[vet[1],], start=c(2000,1), frequency=365)
#plot
p1<-autoplot(models_var) +
  labs(title = paste("VAR (p=",optimal_lag,") Model: MSE_train = ", round(mse_var_train,6), ", MSE_test = ", round(mse_var_test,6)),
       x = "years", y = "prices") +
  scale_color_manual(values = c("black", "red"),
                     labels = c("Ground truth", "VAR")) +
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p1)
#ses plot residuals
p2 <- autoplot(res_fitted) +
  labs(x = "years", y = "residuals", title = "Residuals for DAM prices") +
  scale_color_manual(values = c("VAR               " = "red")) +
  geom_line(aes(color = "VAR               ")) +  # Aggiungo una mappatura del colore
  geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#focus giornaliero
h <- 2020
d1 <- 0
d2 <- 130
models_var_focus <- ts(data.frame(y1=window(ts(matrix_prices[vet[1],(90+optimal_lag):dim(matrix_prices)[2]], start=c(2000,90+optimal_lag), frequency=365), start=c(h,d1), end=c(h,d2)),
                                  y2=window(st_var, start=c(h,d1), end=c(h,d2))), start=c(h,d1), frequency=365)

res_fitted_focus <- window(res_ts, start=c(h,d1), end=c(h,d2))
p1<-autoplot(models_var_focus) +
  labs(title = paste("VAR (",optimal_lag,") on DAM prices at hour 0",sep=""),
       x = "time (days)", y = "prices", color="") +
  scale_color_manual(values = c("black", "red"),
                     labels = c("Ground truth", "VAR")) +
  scale_x_continuous(breaks=seq(2001,2002,by=1/(12)), labels=seq(from = as.Date("2001/01/01"),
                                                                    to = as.Date("2002/01/01"),
                                                                    by = "month")) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
#print(p1)
#ses plot residuals
p2 <- autoplot(res_fitted_focus) +
  labs(x = "years", y = "prices", title = "Residuals for DAM prices") +
  scale_color_manual(values = c("VAR                 " = "red")) +
  geom_line(aes(color = "VAR                 ")) +  # Aggiungo una mappatura del colore
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
#print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#############################################
#SERIE 2
#############################################

coef[1,1:K] <- var_model$varresult$Series.2$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.2$coefficients[((l-1)*K+1):(l*K)]
}
coef
var_fitted <- rep(NA,dim(matrix_prices)[2])
sum <- NA
for (time in (90+optimal_lag-1):(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.2$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted <- var_fitted[(90+optimal_lag):dim(matrix_prices)[2]]
res_fitted <- matrix_prices[vet[2],(90+optimal_lag):dim(matrix_prices)[2]] - var_fitted
st_var <- ts(var_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
res_ts <- ts(res_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
var_test$Hour1 <- st_var
#forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
#fitted <- var_fitted[1:t]
#fitted <- ts(fitted, start=c(2000,1), frequency=365)
#forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_test[2] <- mean(window(res_ts, start=c(2017,238))^2)
abs_fitted <- abs(window(res_ts, start=c(2017,238)))
mae_var_test[2] <- mean(abs_fitted)

#############################################
#SERIE 3
#############################################

coef[1,1:K] <- var_model$varresult$Series.3$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.3$coefficients[((l-1)*K+1):(l*K)]
}
coef
var_fitted <- rep(NA,dim(matrix_prices)[2])
sum <- NA
for (time in (90+optimal_lag-1):(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.3$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted <- var_fitted[(90+optimal_lag):dim(matrix_prices)[2]]
res_fitted <- matrix_prices[vet[3],(90+optimal_lag):dim(matrix_prices)[2]] - var_fitted
st_var <- ts(var_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
res_ts <- ts(res_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
var_test$Hour2 <- st_var
#forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
#fitted <- var_fitted[1:t]
#fitted <- ts(fitted, start=c(2000,1), frequency=365)
#forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_test[3] <- mean(window(res_ts, start=c(2017,238))^2)
abs_fitted <- abs(window(res_ts, start=c(2017,238)))
mae_var_test[3] <- mean(abs_fitted)

#############################################
#SERIE 4
#############################################

coef[1,1:K] <- var_model$varresult$Series.4$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.4$coefficients[((l-1)*K+1):(l*K)]
}
#coef
var_fitted <- rep(NA,dim(matrix_prices)[2])
sum <- NA
for (time in (90+optimal_lag-1):(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.4$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted <- var_fitted[(90+optimal_lag):dim(matrix_prices)[2]]
res_fitted <- matrix_prices[vet[4],(90+optimal_lag):dim(matrix_prices)[2]] - var_fitted
st_var <- ts(var_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
res_ts <- ts(res_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
var_test$Hour3 <- st_var
#forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
#fitted <- var_fitted[1:t]
#fitted <- ts(fitted, start=c(2000,1), frequency=365)
#forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_test[4] <- mean(window(res_ts, start=c(2017,238))^2)
abs_fitted <- abs(window(res_ts, start=c(2017,238)))
mae_var_test[4] <- mean(abs_fitted)

#############################################
#SERIE 5
#############################################

coef[1,1:K] <- var_model$varresult$Series.5$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.5$coefficients[((l-1)*K+1):(l*K)]
}
#coef
var_fitted <- rep(NA,dim(matrix_prices)[2])
sum <- NA
for (time in (90+optimal_lag-1):(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.5$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted <- var_fitted[(90+optimal_lag):dim(matrix_prices)[2]]
res_fitted <- matrix_prices[vet[5],(90+optimal_lag):dim(matrix_prices)[2]] - var_fitted
st_var <- ts(var_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
res_ts <- ts(res_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
var_test$Hour4 <- st_var
#forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
#fitted <- var_fitted[1:t]
#fitted <- ts(fitted, start=c(2000,1), frequency=365)
#forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_test[5] <- mean(window(res_ts, start=c(2017,238))^2)
abs_fitted <- abs(window(res_ts, start=c(2017,238)))
mae_var_test[5] <- mean(abs_fitted)

#focus giornaliero
h <- 2021
d1 <- 266
d2 <- 315
models_var_focus <- ts(data.frame(y1=window(ts(matrix_prices[vet[5],], start=c(2000,1), frequency=365), start=c(h,d1), end=c(h,d2)),
                                  y2=window(st_var, start=c(h,d1), end=c(h,d2))), start=c(h,d1), frequency=365)

res_fitted_focus <- window(res_fitted, start=c(h,d1), end=c(h,d2))


p1<-autoplot(models_var_focus) +
  labs(title = paste("VAR (",optimal_lag,") on DAM prices at hour 4",sep=""),
       x = "time (days)", y = "prices", color="") +
  scale_color_manual(values = c("black", "red"),
                     labels = c("Ground truth", "VAR")) +
  scale_x_continuous(breaks=seq(2021,2022,by=1/(12)), labels=seq(from = as.Date("2021/01/01"),
                                                                 to = as.Date("2022/01/01"),
                                                                 by = "month")) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
#print(p1)
#ses plot residuals
p2 <- autoplot(res_fitted_focus) +
  labs(x = "years", y = "prices", title = "Residuals for DAM prices") +
  scale_color_manual(values = c("VAR(7)               " = "red")) +
  geom_line(aes(color = "VAR(7)               ")) +  # Aggiungo una mappatura del colore
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
#print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#############################################
#SERIE 6
#############################################

coef[1,1:K] <- var_model$varresult$Series.6$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.6$coefficients[((l-1)*K+1):(l*K)]
}
#coef
var_fitted <- rep(NA,dim(matrix_prices)[2])
sum <- NA
for (time in (90+optimal_lag-1):(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.6$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted <- var_fitted[(90+optimal_lag):dim(matrix_prices)[2]]
res_fitted <- matrix_prices[vet[6],(90+optimal_lag):dim(matrix_prices)[2]] - var_fitted
st_var <- ts(var_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
res_ts <- ts(res_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
var_test$Hour5 <- st_var
#forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
#fitted <- var_fitted[1:t]
#fitted <- ts(fitted, start=c(2000,1), frequency=365)
#forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_test[6] <- mean(window(res_ts, start=c(2017,238))^2)
abs_fitted <- abs(window(res_ts, start=c(2017,238)))
mae_var_test[6] <- mean(abs_fitted)

#############################################
#SERIE 7
#############################################

coef[1,1:K] <- var_model$varresult$Series.7$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.7$coefficients[((l-1)*K+1):(l*K)]
}
#coef
var_fitted <- rep(NA,dim(matrix_prices)[2])
sum <- NA
for (time in (90+optimal_lag-1):(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.7$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted <- var_fitted[(90+optimal_lag):dim(matrix_prices)[2]]
res_fitted <- matrix_prices[vet[7],(90+optimal_lag):dim(matrix_prices)[2]] - var_fitted
st_var <- ts(var_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
res_ts <- ts(res_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
var_test$Hour6 <- st_var
#forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
#fitted <- var_fitted[1:t]
#fitted <- ts(fitted, start=c(2000,1), frequency=365)
#forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_test[7] <- mean(window(res_ts, start=c(2017,238))^2)
abs_fitted <- abs(window(res_ts, start=c(2017,238)))
mae_var_test[7] <- mean(abs_fitted)

#############################################
#SERIE 8
#############################################

coef[1,1:K] <- var_model$varresult$Series.8$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.8$coefficients[((l-1)*K+1):(l*K)]
}
coef
var_fitted <- rep(NA,dim(matrix_prices)[2])
sum <- NA
for (time in (90+optimal_lag-1):(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.8$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted <- var_fitted[(90+optimal_lag):dim(matrix_prices)[2]]
res_fitted <- matrix_prices[vet[8],(90+optimal_lag):dim(matrix_prices)[2]] - var_fitted
st_var <- ts(var_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
res_ts <- ts(res_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
var_test$Hour7 <- st_var
#forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
#fitted <- var_fitted[1:t]
#fitted <- ts(fitted, start=c(2000,1), frequency=365)
#forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_test[8] <- mean(window(res_ts, start=c(2017,238))^2)
abs_fitted <- abs(window(res_ts, start=c(2017,238)))
mae_var_test[8] <- mean(abs_fitted)

#############################################
#SERIE 9
#############################################

coef[1,1:K] <- var_model$varresult$Series.9$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.9$coefficients[((l-1)*K+1):(l*K)]
}
#coef
var_fitted <- rep(NA,dim(matrix_prices)[2])
sum <- NA
for (time in (90+optimal_lag-1):(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.9$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted <- var_fitted[(90+optimal_lag):dim(matrix_prices)[2]]
res_fitted <- matrix_prices[vet[9],(90+optimal_lag):dim(matrix_prices)[2]] - var_fitted
st_var <- ts(var_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
res_ts <- ts(res_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
var_test$Hour8 <- st_var
#forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
#fitted <- var_fitted[1:t]
#fitted <- ts(fitted, start=c(2000,1), frequency=365)
#forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_test[9] <- mean(window(res_ts, start=c(2017,238))^2)
abs_fitted <- abs(window(res_ts, start=c(2017,238)))
mae_var_test[9] <- mean(abs_fitted)

#############################################
#SERIE 10
#############################################

coef[1,1:K] <- var_model$varresult$Series.10$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.10$coefficients[((l-1)*K+1):(l*K)]
}
#coef
var_fitted <- rep(NA,dim(matrix_prices)[2])
sum <- NA
for (time in (90+optimal_lag-1):(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.10$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted <- var_fitted[(90+optimal_lag):dim(matrix_prices)[2]]
res_fitted <- matrix_prices[vet[10],(90+optimal_lag):dim(matrix_prices)[2]] - var_fitted
st_var <- ts(var_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
res_ts <- ts(res_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
var_test$Hour9 <- st_var
#forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
#fitted <- var_fitted[1:t]
#fitted <- ts(fitted, start=c(2000,1), frequency=365)
#forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_test[10] <- mean(window(res_ts, start=c(2017,238))^2)
abs_fitted <- abs(window(res_ts, start=c(2017,238)))
mae_var_test[10] <- mean(abs_fitted)

#############################################
#SERIE 11
#############################################

coef[1,1:K] <- var_model$varresult$Series.11$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.11$coefficients[((l-1)*K+1):(l*K)]
}
#coef
var_fitted <- rep(NA,dim(matrix_prices)[2])
sum <- NA
for (time in (90+optimal_lag-1):(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.11$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted <- var_fitted[(90+optimal_lag):dim(matrix_prices)[2]]
res_fitted <- matrix_prices[vet[11],(90+optimal_lag):dim(matrix_prices)[2]] - var_fitted
st_var <- ts(var_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
res_ts <- ts(res_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
var_test$Hour10 <- st_var
#forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
#fitted <- var_fitted[1:t]
#fitted <- ts(fitted, start=c(2000,1), frequency=365)
#forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_test[11] <- mean(window(res_ts, start=c(2017,238))^2)
abs_fitted <- abs(window(res_ts, start=c(2017,238)))
mae_var_test[11] <- mean(abs_fitted)

#############################################
#SERIE 12
#############################################

coef[1,1:K] <- var_model$varresult$Series.12$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.12$coefficients[((l-1)*K+1):(l*K)]
}
coef
var_fitted <- rep(NA,dim(matrix_prices)[2])
sum <- NA
for (time in (90+optimal_lag-1):(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.12$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted <- var_fitted[(90+optimal_lag):dim(matrix_prices)[2]]
res_fitted <- matrix_prices[vet[12],(90+optimal_lag):dim(matrix_prices)[2]] - var_fitted
st_var <- ts(var_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
res_ts <- ts(res_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
var_test$Hour11 <- st_var
#forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
#fitted <- var_fitted[1:t]
#fitted <- ts(fitted, start=c(2000,1), frequency=365)
#forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_test[12] <- mean(window(res_ts, start=c(2017,238))^2)
abs_fitted <- abs(window(res_ts, start=c(2017,238)))
mae_var_test[12] <- mean(abs_fitted)

#############################################
#SERIE 13
#############################################

coef[1,1:K] <- var_model$varresult$Series.13$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.13$coefficients[((l-1)*K+1):(l*K)]
}
coef
var_fitted <- rep(NA,dim(matrix_prices)[2])
sum <- NA
for (time in (90+optimal_lag-1):(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.13$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted <- var_fitted[(90+optimal_lag):dim(matrix_prices)[2]]
res_fitted <- matrix_prices[vet[13],(90+optimal_lag):dim(matrix_prices)[2]] - var_fitted
st_var <- ts(var_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
res_ts <- ts(res_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
var_test$Hour12 <- st_var
#forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
#fitted <- var_fitted[1:t]
#fitted <- ts(fitted, start=c(2000,1), frequency=365)
#forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_test[13] <- mean(window(res_ts, start=c(2017,238))^2)
abs_fitted <- abs(window(res_ts, start=c(2017,238)))
mae_var_test[13] <- mean(abs_fitted)

#############################################
#SERIE 14
#############################################

coef[1,1:K] <- var_model$varresult$Series.14$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.14$coefficients[((l-1)*K+1):(l*K)]
}
coef
var_fitted <- rep(NA,dim(matrix_prices)[2])
sum <- NA
for (time in (90+optimal_lag-1):(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.14$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted <- var_fitted[(90+optimal_lag):dim(matrix_prices)[2]]
res_fitted <- matrix_prices[vet[14],(90+optimal_lag):dim(matrix_prices)[2]] - var_fitted
st_var <- ts(var_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
res_ts <- ts(res_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
var_test$Hour13 <- st_var
#forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
#fitted <- var_fitted[1:t]
#fitted <- ts(fitted, start=c(2000,1), frequency=365)
#forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_test[14] <- mean(window(res_ts, start=c(2017,238))^2)
abs_fitted <- abs(window(res_ts, start=c(2017,238)))
mae_var_test[14] <- mean(abs_fitted)

#############################################
#SERIE 15
#############################################

coef[1,1:K] <- var_model$varresult$Series.15$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.15$coefficients[((l-1)*K+1):(l*K)]
}
#coef
var_fitted <- rep(NA,dim(matrix_prices)[2])
sum <- NA
for (time in (90+optimal_lag-1):(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.15$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted <- var_fitted[(90+optimal_lag):dim(matrix_prices)[2]]
res_fitted <- matrix_prices[vet[15],(90+optimal_lag):dim(matrix_prices)[2]] - var_fitted
st_var <- ts(var_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
res_ts <- ts(res_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
var_test$Hour14 <- st_var
#forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
#fitted <- var_fitted[1:t]
#fitted <- ts(fitted, start=c(2000,1), frequency=365)
#forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_test[15] <- mean(window(res_ts, start=c(2017,238))^2)
abs_fitted <- abs(window(res_ts, start=c(2017,238)))
mae_var_test[15] <- mean(abs_fitted)

#############################################
#SERIE 16
#############################################

coef[1,1:K] <- var_model$varresult$Series.16$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.16$coefficients[((l-1)*K+1):(l*K)]
}
coef
var_fitted <- rep(NA,dim(matrix_prices)[2])
sum <- NA
for (time in (90+optimal_lag-1):(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.16$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted <- var_fitted[(90+optimal_lag):dim(matrix_prices)[2]]
res_fitted <- matrix_prices[vet[16],(90+optimal_lag):dim(matrix_prices)[2]] - var_fitted
st_var <- ts(var_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
res_ts <- ts(res_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
var_test$Hour15 <- st_var
#forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
#fitted <- var_fitted[1:t]
#fitted <- ts(fitted, start=c(2000,1), frequency=365)
#forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_test[16] <- mean(window(res_ts, start=c(2017,238))^2)
abs_fitted <- abs(window(res_ts, start=c(2017,238)))
mae_var_test[16] <- mean(abs_fitted)

#############################################
#SERIE 17
#############################################

coef[1,1:K] <- var_model$varresult$Series.17$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.17$coefficients[((l-1)*K+1):(l*K)]
}
#coef
var_fitted <- rep(NA,dim(matrix_prices)[2])
sum <- NA
for (time in (90+optimal_lag-1):(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.17$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted <- var_fitted[(90+optimal_lag):dim(matrix_prices)[2]]
res_fitted <- matrix_prices[vet[17],(90+optimal_lag):dim(matrix_prices)[2]] - var_fitted
st_var <- ts(var_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
res_ts <- ts(res_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
var_test$Hour16 <- st_var
#forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
#fitted <- var_fitted[1:t]
#fitted <- ts(fitted, start=c(2000,1), frequency=365)
#forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_test[17] <- mean(window(res_ts, start=c(2017,238))^2)
abs_fitted <- abs(window(res_ts, start=c(2017,238)))
mae_var_test[17] <- mean(abs_fitted)


#focus giornaliero
h <- 2020
d1 <- 143
#d1 <- 200
d2 <- d1+150
models_var_focus <- ts(data.frame(y1=window(ts(matrix_prices[vet[17],(90+optimal_lag):dim(matrix_prices)[2]], start=c(2000,90+optimal_lag), frequency=365), start=c(h,d1), end=c(h,d2)),
                                  y2=window(st_var, start=c(h,d1), end=c(h,d2))), start=c(h,d1), frequency=365)

res_fitted_focus <- window(res_ts, start=c(h,d1), end=c(h,d2))

p1<-autoplot(models_var_focus) +
  labs(title = paste("VAR (",optimal_lag,") on DAM prices at hour 17",sep=""),
       x = "time (days)", y = "prices", color="") +
  scale_color_manual(values = c("black", "red"),
                     labels = c("Ground truth", "VAR")) +
  scale_x_continuous(breaks=seq(h,h+1,by=1/(12)), labels=seq(from = as.Date("2021/01/01"),
                                                                 to = as.Date("2022/01/01"),
                                                                 by = "month")) +
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
#print(p1)
#ses plot residuals
p2 <- autoplot(res_fitted_focus) +
  labs(x = "years", y = "prices", title = "Residuals for DAM prices") +
  scale_color_manual(values = c("VAR(7)               " = "red")) +
  geom_line(aes(color = "VAR(7)               ")) +  # Aggiungo una mappatura del colore
  #geom_vline(xintercept=2017.6, linetype = "dashed", color = "black") +
  theme_bw()
#print(p2)
#grid with forecasts and corresponding residuals
gridExtra::grid.arrange(p1, p2, ncol = 1)

#############################################
#SERIE 18
#############################################

coef[1,1:K] <- var_model$varresult$Series.18$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.18$coefficients[((l-1)*K+1):(l*K)]
}
#coef
var_fitted <- rep(NA,dim(matrix_prices)[2])
sum <- NA
for (time in (90+optimal_lag-1):(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.18$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted <- var_fitted[(90+optimal_lag):dim(matrix_prices)[2]]
res_fitted <- matrix_prices[vet[18],(90+optimal_lag):dim(matrix_prices)[2]] - var_fitted
st_var <- ts(var_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
res_ts <- ts(res_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
var_test$Hour17 <- st_var
#forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
#fitted <- var_fitted[1:t]
#fitted <- ts(fitted, start=c(2000,1), frequency=365)
#forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_test[18] <- mean(window(res_ts, start=c(2017,238))^2)
abs_fitted <- abs(window(res_ts, start=c(2017,238)))
mae_var_test[18] <- mean(abs_fitted)

#############################################
#SERIE 19
#############################################

coef[1,1:K] <- var_model$varresult$Series.19$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.19$coefficients[((l-1)*K+1):(l*K)]
}
coef
var_fitted <- rep(NA,dim(matrix_prices)[2])
sum <- NA
for (time in (90+optimal_lag-1):(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.19$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted <- var_fitted[(90+optimal_lag):dim(matrix_prices)[2]]
res_fitted <- matrix_prices[vet[19],(90+optimal_lag):dim(matrix_prices)[2]] - var_fitted
st_var <- ts(var_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
res_ts <- ts(res_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
var_test$Hour18 <- st_var
#forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
#fitted <- var_fitted[1:t]
#fitted <- ts(fitted, start=c(2000,1), frequency=365)
#forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_test[19] <- mean(window(res_ts, start=c(2017,238))^2)
abs_fitted <- abs(window(res_ts, start=c(2017,238)))
mae_var_test[19] <- mean(abs_fitted)

#############################################
#SERIE 20
#############################################

coef[1,1:K] <- var_model$varresult$Series.20$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.20$coefficients[((l-1)*K+1):(l*K)]
}
#coef
var_fitted <- rep(NA,dim(matrix_prices)[2])
sum <- NA
for (time in (90+optimal_lag-1):(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.20$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted <- var_fitted[(90+optimal_lag):dim(matrix_prices)[2]]
res_fitted <- matrix_prices[vet[20],(90+optimal_lag):dim(matrix_prices)[2]] - var_fitted
st_var <- ts(var_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
res_ts <- ts(res_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
var_test$Hour19 <- st_var
#forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
#fitted <- var_fitted[1:t]
#fitted <- ts(fitted, start=c(2000,1), frequency=365)
#forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_test[20] <- mean(window(res_ts, start=c(2017,238))^2)
abs_fitted <- abs(window(res_ts, start=c(2017,238)))
mae_var_test[20] <- mean(abs_fitted)

#############################################
#SERIE 21
#############################################

coef[1,1:K] <- var_model$varresult$Series.21$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.21$coefficients[((l-1)*K+1):(l*K)]
}
#coef
var_fitted <- rep(NA,dim(matrix_prices)[2])
sum <- NA
for (time in (90+optimal_lag-1):(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.21$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted <- var_fitted[(90+optimal_lag):dim(matrix_prices)[2]]
res_fitted <- matrix_prices[vet[21],(90+optimal_lag):dim(matrix_prices)[2]] - var_fitted
st_var <- ts(var_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
res_ts <- ts(res_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
var_test$Hour20 <- st_var
#forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
#fitted <- var_fitted[1:t]
#fitted <- ts(fitted, start=c(2000,1), frequency=365)
#forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_test[21] <- mean(window(res_ts, start=c(2017,238))^2)
abs_fitted <- abs(window(res_ts, start=c(2017,238)))
mae_var_test[21] <- mean(abs_fitted)

#############################################
#SERIE 22
#############################################

coef[1,1:K] <- var_model$varresult$Series.22$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.22$coefficients[((l-1)*K+1):(l*K)]
}
#coef
var_fitted <- rep(NA,dim(matrix_prices)[2])
sum <- NA
for (time in (90+optimal_lag-1):(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.22$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted <- var_fitted[(90+optimal_lag):dim(matrix_prices)[2]]
res_fitted <- matrix_prices[vet[22],(90+optimal_lag):dim(matrix_prices)[2]] - var_fitted
st_var <- ts(var_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
res_ts <- ts(res_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
var_test$Hour21 <- st_var
#forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
#fitted <- var_fitted[1:t]
#fitted <- ts(fitted, start=c(2000,1), frequency=365)
#forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_test[22] <- mean(window(res_ts, start=c(2017,238))^2)
abs_fitted <- abs(window(res_ts, start=c(2017,238)))
mae_var_test[22] <- mean(abs_fitted)

#############################################
#SERIE 23
#############################################

coef[1,1:K] <- var_model$varresult$Series.23$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.23$coefficients[((l-1)*K+1):(l*K)]
}
#coef
var_fitted <- rep(NA,dim(matrix_prices)[2])
sum <- NA
for (time in (90+optimal_lag-1):(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.23$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted <- var_fitted[(90+optimal_lag):dim(matrix_prices)[2]]
res_fitted <- matrix_prices[vet[23],(90+optimal_lag):dim(matrix_prices)[2]] - var_fitted
st_var <- ts(var_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
res_ts <- ts(res_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
var_test$Hour22 <- st_var
#forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
#fitted <- var_fitted[1:t]
#fitted <- ts(fitted, start=c(2000,1), frequency=365)
#forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_test[23] <- mean(window(res_ts, start=c(2017,238))^2)
abs_fitted <- abs(window(res_ts, start=c(2017,238)))
mae_var_test[23] <- mean(abs_fitted)

#############################################
#SERIE 24
#############################################

coef[1,1:K] <- var_model$varresult$Series.24$coefficients[1:K]
for (l in 2:p){
  coef[l,1:K] <- var_model$varresult$Series.24$coefficients[((l-1)*K+1):(l*K)]
}
#coef
var_fitted <- rep(NA,dim(matrix_prices)[2])
sum <- NA
for (time in (90+optimal_lag-1):(dim(matrix_prices)[2]-1)){
  sum <- var_model$varresult$Series.24$coefficients[n_par/K]
  for (k in 1:p){
    for (j in 1:K){
      sum <- sum + coef[k,j]*matrix_prices[vet[j],time-k+1]
    }
  }
  var_fitted[time+1] <- sum
}
var_fitted <- var_fitted[(90+optimal_lag):dim(matrix_prices)[2]]
res_fitted <- matrix_prices[vet[24],(90+optimal_lag):dim(matrix_prices)[2]] - var_fitted
st_var <- ts(var_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
res_ts <- ts(res_fitted, start=c(2000,(90+optimal_lag)), frequency=365)
var_test$Hour23 <- st_var
#forecast <- var_fitted[(t+1):dim(matrix_prices)[2]]
#fitted <- var_fitted[1:t]
#fitted <- ts(fitted, start=c(2000,1), frequency=365)
#forecast <- ts(forecast, end=c(2021,365), frequency=365)
mse_var_test[24] <- mean(window(res_ts, start=c(2017,238))^2)
abs_fitted <- abs(window(res_ts, start=c(2017,238)))
mae_var_test[24] <- mean(abs_fitted)



###########################à
#estrazione test

write.csv(var_test, file = "VSCODE/PyG/DAM/var_test.csv", row.names = FALSE)


################################################
# estrazione mse



errors_var <- data.frame(y1=mse_var_test, y2=mae_var_test)
print(errors_var)

write.csv(errors_var, file = "VSCODE/PyG/DAM/errors_var.csv", row.names = FALSE)


###############

df_compare <- data.frame(y1=window(st_var, start=c(h,d1), end=c(h,d2)),
                         y2=res_fitted_focus,
                         y3=window(st_var1, start=c(h,d1), end=c(h,d2)),
                         y2=res_fitted_focus1)
write.csv(df_compare, "VSCODE/PyG/DAM/var_compare.csv", row.names = FALSE)
