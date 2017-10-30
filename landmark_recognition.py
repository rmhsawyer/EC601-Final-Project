from PIL import Image, ImageDraw, ImageColor
import face_recognition

# Load the jpg file into a numpy array
image = face_recognition.load_image_file("sophie.jpg")

# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image)

for face_landmarks in face_landmarks_list:
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image, 'RGBA')
    red=ImageColor.getrgb("red")
    green=ImageColor.getrgb("green")
    blue=ImageColor.getrgb("blue")
    yellow=ImageColor.getrgb("yellow")

    d.line(face_landmarks['nose_bridge'], fill=yellow, width=5)
    d.line(face_landmarks['nose_tip'], fill=yellow, width=5)

    d.line(face_landmarks['chin'], fill=ImageColor.getrgb("white"), width=5)

    # Make the eyebrows into a nightmare
    #d.polygon(face_landmarks['left_eyebrow'], fill=red)
    #d.polygon(face_landmarks['right_eyebrow'], fill=red)
    d.line(face_landmarks['left_eyebrow'], fill=red, width=5)
    d.line(face_landmarks['right_eyebrow'], fill=red, width=5)

    # Gloss the lips
    #d.polygon(face_landmarks['top_lip'], fill=blue)
    #d.polygon(face_landmarks['bottom_lip'], fill=blue)
    d.line(face_landmarks['top_lip'], fill=blue, width=8)
    d.line(face_landmarks['bottom_lip'], fill=blue, width=8)

    # Sparkle the eyes
    #d.polygon(face_landmarks['left_eye'], fill=green)
    #d.polygon(face_landmarks['right_eye'], fill=green)
    

    # Apply some eyeliner
    d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=green, width=6)
    d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=green, width=6)
    
    pil_image.show()
    pil_image.save("sophie_ugly.jpg")


    
