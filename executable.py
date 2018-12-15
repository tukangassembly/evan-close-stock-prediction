import model

data, algorithm = None, None
print("1 : Waskita")
print("2 : BCA")
print("3 : CIMB")
while(data is None):
   user_selection = int(input("Select a dataset : "))
   if user_selection == 1:
       data = "WSKT.JK.csv"
   elif user_selection == 2:
       data = "BBCA.JK.csv"
   elif user_selection == 3:
       data = "BNGA.JK.csv"
   else:
       print("Please select a valid option!")
       continue


print("1 : Linear Regression")
print("2 : Random Forest")
print("3 : Support Vector Machine")
while(algorithm is None):
   selected_algorithm = int(input("Select an algorithm : "))
   if selected_algorithm == 1:
       algorithm = selected_algorithm
       model.linearRegressor(data)
   elif selected_algorithm == 2:
       algorithm = selected_algorithm
       model.randForestRegressor(data)
   elif selected_algorithm == 3:
       algorithm = selected_algorithm
       model.SVM(data)
   else:
       print("Please select a valid option!")
       continue
