rm(list = ls())

library(outliers)

set.seed(16)
#not needed but I added this to continue practicing adding a seed to my code blocks
crime_data <- read.table(
  "C:/Users/mstout/OneDrive - AANP/Documents/Workspace.MAIN/edX/GTx.ISYE6501/data/Homework3_ISYE_6501/data 5.1/uscrime.txt",
  header = TRUE
)

print(dim(crime_data))

g <- grubbs.test(na.omit(crime_data$Crime))
#type = 20 was not working for .Rmd file,
#so I had to use this after already knowing the higher value was the possible outlier
print(g)

# ---------------------------------------------------------------------------------------------------------------------------------------------------

path <- "C:/Users/mstout/OneDrive - AANP/Documents/Workspace.MAIN/edX/GTx.ISYE6501/data/Homework3_ISYE_6501/data 6.2/temps.txt"
w <- read.delim(path, sep = "\t", header = TRUE, check.names = FALSE)

print(dim(w))

years <- names(w)[names(w) != "DAY"]
d <- as.Date(paste0("2000-", w$DAY), format = "%Y-%d-%b")
mmdd <- format(d, "%m-%d")

L <- do.call(rbind, lapply(years, function(y) {
  data.frame(
    year = as.integer(y),
    date = as.Date(paste0(y, "-", mmdd)),
    tmax = as.numeric(w[[y]])
  )
}))
L <- L[order(L$year, L$date), ]

one_year <- function(z) {
  z <- z[order(z$date), ]
  x <- z$tmax
  s <- cumsum(x - mean(x, na.rm = TRUE))
  k <- which.max(s)
  data.frame(
    year = z$year[1],
    end_date = z$date[k],
    end_doy = as.integer(format(z$date[k], "%j")),
    mean_before = mean(x[1:k], na.rm = TRUE),
    mean_after = mean(x[(k + 1):length(x)], na.rm = TRUE)
  )
}

R <- do.call(rbind, lapply(split(L, L$year), one_year))
R$drop_after <- R$mean_after - R$mean_before
R <- R[order(R$year), ]

A <- aggregate(tmax ~ year, L, mean)
A <- A[order(A$year), ]

Z <- split(L, L$year)[["2014"]]
Z <- Z[order(Z$date), ]
S <- cumsum(Z$tmax - mean(Z$tmax, na.rm = TRUE))
k2014 <- which.max(S)

print(summary(lm(mean_before ~ year, data = R)))
print(summary(lm(end_doy ~ year, data = R)))

par(mfrow = c(2, 2))

plot(R$year, R$end_doy, xlab = "Year", ylab = "End of summer day of year")
abline(lm(end_doy ~ year, data = R))

plot(R$year, R$mean_before, xlab = "Year", ylab = "Mean before end date")
abline(lm(mean_before ~ year, data = R))

plot(A$year, A$tmax, xlab = "Jul to Oct mean high", ylab = "Temperature")

plot(Z$date, S, type = "l", xlab = "Date", ylab = "CUSUM")
abline(v = Z$date[k2014], lty = 2)

print(dim(w))
print(dim(L))
print(dim(R))
print(head(R, 20)) #just want to ensure it all loaded as expected