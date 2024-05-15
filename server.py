import os
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
app = Flask(__name__)
sign_data_path = os.path.join(os.getcwd(),"sign_data", "users")
# Initial users dictionary. Picked three folders from inside the test dataset
users = {
    1: {
        "name": "Saad",
        "user_id": 1,
        "signatures": []
    },
    2: {
        "name": "Hashir",
        "user_id": 2,
        "signatures": []
    },
    3: {
        "name": "Imad",
        "user_id": 3,
        "signatures": []
    },
}
# Hardcoded usernames and passwords
user_credentials = {
    "admin123": "letmein"
}
global accuracy


# Function to read images from a folder and store them in user signatures
def read_user_signatures(user_id, folder_path):

  if user_id not in users:
    print(f"User with ID {user_id} not found in the dictionary.")
    return

  # Get a list of image files within the folder
  image_files = [f for f in os.listdir(folder_path)]
#   print(image_files)
  # Process each image file (modify as needed)
  for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)

    # Read the image using Image from PIL 
    img = Image.open(image_path)
    img = img.convert('L')
    img = img.resize((64, 64))
    
    # Append the processed image data (replace with your actual data)
    
    users[user_id]["signatures"].append(img)

# Loop through user folders and process signatures
for user_id, user_data in users.items():
  folder_path = os.path.join(sign_data_path, str(user_id))
  if os.path.isdir(folder_path):
    
    read_user_signatures(user_id, folder_path)

print(users)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/admin")
def admin():
    user = request.args.get("Username")
    password = request.args.get("Password")
    # Check if the entered username and password match the hardcoded credentials
    if user in user_credentials and user_credentials[user] == password:
        return render_template("admin.html")
    return render_template("admin_not_found.html")

@app.route("/add_user", methods=["GET", "POST"])
def add_user():
    if request.method == "POST":
        user_id = int(request.form.get("userId"))
        user_name = request.form.get("userName")
        # Add the new user to the dictionary
        users[user_id] = user_name
        image_file = request.files["image"]

        # Save the file to a temporary location
        image_file_path = os.path.join("/tmp", image_file.filename)
        image_file.save(image_file_path)

        # Get the content length after saving the file
        content_length = os.path.getsize(image_file_path)

        # Process the image here
        # print(user_id)
        # print(user_name)
        # print("File:", image_file)
        # print(image_file.content_type)
        return redirect(url_for("detection"))
    return render_template("add_user.html")

@app.route("/detection")
def detection():
    # print(users.items())
    
    
    
    
    return render_template("detection.html", users=users)

@app.route("/process_image", methods=["GET", "POST"])
def process_image():
    
    if request.method == "POST":
        user_id = request.form.get("userId")
        image_file = request.files["image"]

        # Save the file to a temporary location
        image_file_path = os.path.join("/tmp", image_file.filename)
        image_file.save(image_file_path)

        # Get the content length after saving the file
        content_length = os.path.getsize(image_file_path)

        # Process the image here
        global accuracy 
        accuracy = 98
        # Check for trigger signal from hidden field
        trigger_model = request.form.get("triggerModel")
        print(trigger_model)
        if trigger_model == "true":
            # Call the model function (assuming it processes the image)
            print(model(5,5))  # Replace with actual processing logic
                           
        return redirect(url_for("process_image", accuracy=accuracy))
    
    # Handle GET request (optional)
    # For example, you can render a form for uploading images
        
    return render_template("process_image.html", accuracy=accuracy)

def model(x,y):
    return x+y

if __name__ == "__main__":
    app.run("localhost", 4448)
