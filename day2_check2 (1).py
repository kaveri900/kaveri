
total_bill = float(input("Enter the total bill amount: "))
people = int(input("Enter the number of people: "))
share = total_bill / people
# Print result
print("Total Bill:", total_bill)
print("Each person pays:", share)
# Bonus: check data types
print("\nData Types:")
print("Type of total_bill:", type(total_bill))
print("Type of people:", type(people))
print("Type of share:", type(share))
