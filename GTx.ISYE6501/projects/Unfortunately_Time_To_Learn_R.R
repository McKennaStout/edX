plot(1:10) # How to create a graph with numbers from 1:10 on both the x and y axis


# Creating Variables in R
name <- "McKenna"
age <- 27
print(name)
print(age)


#Data Types
# numeric
x <- 10.5
class(x)

# integer
x <- 1000L
class(x)

# complex
x <- 9i + 3
class(x)

# character/string
x <- "R is exciting"
class(x)

# logical/boolean
x <- TRUE
class(x)

# There are three number types in R:
#numeric
A numeric data type is the most common type in R, and contains any number with or without a decimal, like: 10.5, 55, 787
#integer
Integers are numeric data without decimals. This is used when you are certain that you will never create a variable that should contain decimals. To create an integer variable, you must use the letter L after the integer value
#complex
A complex number is written with an "i" as the imaginary part

:	#Creates a series of numbers in a sequence	x <- 1:10
%in%	#Find out if an element belongs to a vector	x %in% y
%*%	#Matrix Multiplication	x <- Matrix1 %*% Matrix2

#if v else if v if else
a <- 200
b <- 33

if (b > a) {
  print("b is greater than a")
} else if (a == b) {
  print("a and b are equal")
} else {
  print("a is greater than b")
}

# you can also have nested if statements
x <- 41

if (x > 10) {
  print("Above ten")
  if (x > 20) {
    print("and also above 20!")
  } else {
    print("but not above 20.")
  }
} else {
  print("below 10.")
}

# The & symbol (and) is a logical operator, and is used to combine conditional statements
# The | symbol (or) is a logical operator, and is used to combine conditional statements

# With the (while) loop we can execute a set of statements as long as a condition is TRUE
i <- 1
while (i < 6) {
  print(i)
  i <- i + 1
}

# With the break statement, we can stop the loop even if the while condition is TRUE
i <- 1
while (i < 6) {
  print(i)
  i <- i + 1
  if (i == 4) {
    break
  }
}

# With the next statement, we can skip an iteration without terminating the loop
i <- 0
while (i < 6) {
  i <- i + 1
  if (i == 3) {
    next
  }
  print(i)
}

# Practical Example
dice <- 1
while (dice <= 6) {
  if (dice < 6) {
    print("No Yahtzee")
  } else {
    print("Yahtzee!")
  }
  dice <- dice + 1
}

# A for loop is used for iterating over a sequence
for (x in 1:10) {
  print(x)
}

#Print every item in a list
fruits <- list("apple", "banana", "cherry")

for (x in fruits) {
  print(x)
}

# practical example
dice <- 1:6

for(x in dice) {
  if (x == 6) {
    print(paste("The dice number is", x, "Yahtzee!"))
  } else {
    print(paste("The dice number is", x, "Not Yahtzee"))
  }
}
# If the loop reaches the values ranging from 1 to 5, it prints "No Yahtzee" and its number. When it reaches the value 6, it prints "Yahtzee!" and its number.

# nested loops
adj <- list("red", "big", "tasty")

fruits <- list("apple", "banana", "cherry")
  for (x in adj) {
    for (y in fruits) {
      print(paste(x, y))
  }
}

# to make and call your function
my_function <- function() {
  print("Hello World!")
}

my_function() # call the function named my_function

# arguments

my_function <- function(fname) {
  paste(fname, "Griffin")
}

my_function("Peter")
my_function("Lois")
my_function("Stewie")

# Arguments are specified after the function name, inside the parentheses. You can add as many arguments as you want, just separate them with a comma.

# Default Parameter Value
my_function <- function(country = "Norway") {
  paste("I am from", country)
}

my_function("Sweden")
my_function("India")
my_function() # will get the default value, which is Norway
my_function("USA")

# Think of df as a cabinet.
Columns are folders inside it.
if you have a dataframe named, df:
df$R1 means:
Open the cabinet called df, take out the folder labeled R1.

# base model to input kknn()
kknn(formula, train, test, k, ...)

# more advanced way to use kknn()
kknn(
  formula,
  train,
  test,
  k = 7,
  distance = 2,
  kernel = "optimal",
  scale = TRUE
)
