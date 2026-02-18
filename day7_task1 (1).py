# Ask user for input
name = input("Enter your name: ")
goal = input("Enter your daily goal: ")

# Open file in append mode
with open("journal.txt", "a") as file:
    file.write(f"Name: {name}, Daily Goal: {goal}\n")

print("Entry saved successfully!")
