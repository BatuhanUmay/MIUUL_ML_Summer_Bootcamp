string = "hi my name is john and i am learning python"
my = ""

for i in range(len(string)):
    if i % 2 == 0:
        my += string[i].upper()
    else:
        my += string[i]

print(my)

########################################################################

string = "hi my name is john and i am learning python"
my = ""
for idx, data in enumerate(string):
    if idx % 2 == 0:
        my += data.upper()
    else:
        my += data
print(my)