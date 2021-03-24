import cv2
import pickle

fc = cv2.CascadeClassifier("cas\data\haarcascade_frontalface_alt2.xml")
recognizer =  cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainnerV2.yml")
i=30
labels = {}
with open("labels.pickle" , 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}


video_capture = cv2.VideoCapture(0)

    
while True:

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
 
    faces = fc.detectMultiScale(gray, scaleFactor=1.5 , minNeighbors=5)
    for(x, y , w, h) in faces:
       
        roi_gray = gray[y:(y+h+80) , x:x+w]
        #roi_color = frame[y:y+h , x:x+w]        
        id_ , conf = recognizer.predict(roi_gray)
        #print(conf)
        if conf>=45 and conf <= 140:
            
            
           # print(labels[id_])
            font = cv2.FONT_HERSHEY_COMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame , name , (x ,y ), font , 1 , color , stroke , cv2.LINE_AA)
        
        #cv2.imshow('image',img_item)
        color = (255 ,0 , 0 )
        stroke = 2
        endX = x + w
        endY = y + h+60

        cv2.rectangle(frame , (x,y),(endX,endY),color,stroke)
        
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        s = ".png"
        img_item = str(i) + s
        print(roi_gray)
        cv2.imwrite(img_item,roi_gray)
        i += 1
    # exit if you press key `q`
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#When everything is done, release the capture

video_capture.release()
cv2.destroyAllWindows()
