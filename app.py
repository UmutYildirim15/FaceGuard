import glob
import random
import time
import tkinter as tk
from collections import deque
from tkinter import messagebox, simpledialog
import cv2
import threading
import os
from PIL import Image, ImageTk, ImageDraw
from ftplib import FTP
from tqdm import tqdm
import updated_simple_facerec
import shutil
import json
import base64


def resize_camera_image(frame, width, height):
    return cv2.resize(frame, (width, height))


class FaceRecognitionGUI:
    def __init__(self, root):
        # FTP sunucusuna bağlanma



        self.ftp = FTP('192.168.1.2')
        self.ftp.login(user='umuty', passwd='15022001umut')
        # print(self.ftp.pwd())

        self.custom_stack = CustomStack()

        self.success_screen = None
        self.stop_checking = False
        self.success_screen_open = False

        self.success_screen_open = False
        self.last_person_identity = None

        self.faculty = None
        self.name = None
        self.surname = None
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("800x600+0+0")
        self.root.configure(bg="#164B8B")
        self.frame = tk.Frame(root, bg="#164B8B")
        self.frame.pack(expand=True, fill="both")

        self.button_frame = tk.Frame(self.frame, bg="#164B8B")
        self.button_frame.pack(expand=True, fill="both")
        self.process_this_frame = True



        #self.add_user_button = tk.Button(self.button_frame, text="Add New User", command=self.add_new_user, height=2,
        #                                 width=12)
        #self.add_user_button.pack(side="top", padx=10, pady=10)

        self.check_user_button = tk.Button(self.button_frame, text="Check User", command=self.updated_check_user,
                                           height=2,
                                           width=12)
        self.check_user_button.pack(side="top", padx=10, pady=10)

        self.quit_button = tk.Button(self.button_frame, text="Quit", command=self.quit_app, height=2, width=12)
        self.quit_button.pack(side="top", padx=10, pady=10)

        self.camera_label = tk.Label(self.frame)
        self.camera_label.pack(pady=10)

        self.cap = None  # Initialize the video capture object
        self.camera_opened = False

        # self.updated_check_user()

    def retrieve_images_from_ftp(self):
        target_directory = 'D:\\TEDFaceRecognition\\data\\images_temp\\'  # Hedef klasörün yolu
        # checked_files = set()  # İndirilen dosyaları takip etmek için bir set oluşturuldu.

        # Sunucudaki fotoların listesini alma
        file_list = self.ftp.nlst('*.jpg')
        if file_list:
            # En son eklenen foto ismini bulmak
            latest_file = max(file_list)

            for filename in os.listdir(target_directory):
                file_path = os.path.join(target_directory, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Hata, images_temp boş: {e}")
            # print("Latest file2: " + latest_file)
            # En son eklenen dosyayı indirme işlemi
            with open(target_directory + latest_file, 'wb') as file:
                self.ftp.retrbinary(f'RETR {latest_file}', file.write)
            # self.ftp.delete(latest_file)
            # Son foto hariç tüm fotolar sunucudan silinir. Son foto return edilir.
            for photo in file_list:
                if photo != latest_file:
                    self.ftp.delete(photo)
            return latest_file

    def retrieve_images_from_ftp1(self):
        target_directory = 'D:\\TEDFaceRecognition\\data\\images_temp\\'  # Hedef klasörün yolu
        # checked_files = set()  # İndirilen dosyaları takip etmek için bir set oluşturuldu.

        # Sunucudaki dosyaların listesini alma
        file_list = self.ftp.nlst()
        # En son eklenen dosyanın ismini bulmak
        latest_file = max(file_list, key=lambda x: self.ftp.sendcmd('MDTM ' + x)[4:].strip())
        # print("Latest file1: "+latest_file)

        for filename in os.listdir(target_directory):
            file_path = os.path.join(target_directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Hata, images_temp boş: {e}")
        # print("Latest file2: " + latest_file)
        # En son eklenen dosyayı indirme işlemi
        with open(target_directory + latest_file, 'wb') as file:
            self.ftp.retrbinary(f'RETR {latest_file}', file.write)

    def add_new_user(self):
        self.name = simpledialog.askstring("Add New User", "Name:")
        self.surname = simpledialog.askstring("Add New User", "Surname:")
        self.faculty = simpledialog.askstring("Add New User", "Faculty:")

        if not self.name or not self.surname or not self.faculty:
            messagebox.showerror("Error", "Please enter valid information.")
        else:
            messagebox.showinfo("Info", "Please wait for face detection...")

    def stop_user_checking(self):
        self.stop_checking = True
        if self.cap is not None:
            self.cap.release()  # Release the camera
            self.camera_opened = False
        if self.success_screen_open and self.success_screen and self.success_screen.winfo_exists():
            self.success_screen.destroy()  # Destroy the success screen if it's open and exists
            self.success_screen_open = False
        self.camera_label.pack_forget()

    def updated_check_user(self):
        if not self.camera_opened:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Warning!", "Camera cannot be opened.")
                return
            self.camera_opened = True
        self.camera_label.pack()
        # messagebox.showinfo("Info", "Please wait for face recognition...")
        stop_button = tk.Button(self.success_screen, text="Stop", command=self.stop_user_checking)
        stop_button.pack(side=tk.TOP, pady=10)

        self.retrieve_images_from_ftp()
        recognition_thread = threading.Thread(target=self.updated_run_face_recognition)
        recognition_thread.start()
        # contain npy for embeddings and registration photos

    def updated_run_face_recognition(self):

        ftp_images_directory = 'D:\\TEDFaceRecognition\\data\\images_temp'
        directory = 'data'

        weights = os.path.join(directory, "models", "face_detection_yunet_2023mar.onnx")
        face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
        face_detector.setScoreThreshold(0.87)

        weights = os.path.join(directory, "models", "face_recognizer_fast.onnx")
        face_recognizer = cv2.FaceRecognizerSF_create(weights, "")

        dictionary = {}
        types = ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG')
        files = []
        for a_type in types:
            files.extend(glob.glob(os.path.join(directory, 'images', a_type)))

        files = list(set(files))

        for file in tqdm(files):
            image = cv2.imread(file)
            features, faces = updated_simple_facerec.recognize_face(image, face_detector, face_recognizer, file)
            if faces is None:
                continue
            user_id = file  # Store the file path as the ID
            dictionary[user_id] = features[0]  # Use the first feature as the representative for the ID

        # self.retrieve_images_from_ftp()
        # ftp_image_files = os.listdir(ftp_images_directory)
        while True:
            latest_file = self.retrieve_images_from_ftp()
            ftp_image_files = os.listdir(ftp_images_directory)
            for ftp_image_file in ftp_image_files:
                # print(ftp_image_files)
                # self.retrieve_images_from_ftp()
                if ftp_image_file is None:
                    break
                ftp_image_path = os.path.join(ftp_images_directory, ftp_image_file)
                # print(ftp_image_path)
                image = cv2.imread(ftp_image_path)
                features, faces = updated_simple_facerec.recognize_face(image, face_detector, face_recognizer,
                                                                        ftp_image_path)
                if faces is None:
                    continue

                for idx, (face, feature) in enumerate(zip(faces, features)):
                    result, user = updated_simple_facerec.match(face_recognizer, feature, dictionary)
                    id_name, score = user if result else (f"unknown_{idx}", 0.0)

                    #current_time = time.time()

                    if result:
                        parcalar = id_name.split("\\")
                        username = parcalar[-1]
                        parcalar = username.split(".jpg")
                        username = parcalar[0]
                        username = username.replace(" ", "_")
                        self.custom_stack.pop_expired(time.time())
                        #print("Tespit edilen kişi : " + username)
                        #print(self.custom_stack.get_last_element_name())
                        if self.custom_stack.is_user_available(username):
                            print(username)
                            self.custom_stack.push(username, time.time())
                            self.send_json_toFtp(username, id_name, 1, latest_file, self.custom_stack.get_assigned_turnstile_id(username), recognizationResult=True)
                            self.show_success_screen(f"D:\\TEDFaceRecognition\\{id_name}", username,
                                                     "Computer Engineering")
                            # print("Tespit edilen kişi : " + id_name)
                        # print("stack son eleman:" + self.custom_stack.get_last_element_name())
                    # if result:    # Kullanıcı daha önce tespit edilmişse (5 saniye) tekrar tespit etme!
                    # self.send_json_toFtp(id_name, 1, recognizationResult=True)
                    # print("Tespit edilen kişi : "+id_name)
                    # self.custom_stack.pop_expired()
                    # self.custom_stack.push(id_name)
                    # self.show_success_screen(f"D:\\TEDFaceRecognition\\{id_name}", "Umut Yıldırım",
                    #                         "Computer Engineering")
                    else:
                        parcalar = id_name.split("\\")
                        username = parcalar[-1]
                        parcalar = username.split(".jpg")
                        username = parcalar[0]
                        username = username.replace(" ", "_")
                        self.send_json_toFtp(username, None, 1, latest_file, -1, recognizationResult=False)
                    self.custom_stack.pop_expired(time.time())

                    # print("stack son eleman:" + self.custom_stack.get_last_element_name())
                    # print(result)

                    # self.custom_stack.pop_expired(current_time)
                    # frame_with_roi = image.copy()
                    # height, width, _ = frame_with_roi.shape
                    # top_boundary = int(height * 0.25)
                    # cv2.rectangle(frame_with_roi, (0, top_boundary), (width, top_boundary), (255, 0, 0), 2, cv2.LINE_AA)

                    # img = Image.fromarray(cv2.cvtColor(frame_with_roi, cv2.COLOR_BGR2RGB))
                    # img = ImageTk.PhotoImage(image=img)
                    # self.camera_label.config(image=img)
                    # self.camera_label.image = img

            # self.cap.release() # Burada kamera serbest bırakılmamalıdır, çünkü FTP sunucusu ile ilgili bir işlem yapılmadı.

    def send_json_toFtp(self, username, id_name, gateID, latest_file, turnstile_id, recognizationResult):

        # self.ftp.delete("result.json")
        # Recognization result'a göre JSON dosyası oluşturun
        if recognizationResult:
            # Görüntüyü base64 formatına dönüştürme
            with open(f"D:\\TEDFaceRecognition\\{id_name}",
                      "rb") as image_file:  ## Bu görsel farklı bir isimle (result_time gibi) kaydedilebilir encoding yerine.
                encoded_image_show = base64.b64encode(image_file.read()).decode("utf-8")
            with open(f"D:\\TEDFaceRecognition\\data\\images_temp\\{latest_file}",
                      "rb") as image_file:  ## Bu görsel farklı bir isimle (result_time gibi) kaydedilebilir encoding yerine.
                encoded_image_detected = base64.b64encode(image_file.read()).decode("utf-8")

            result = {
                "recognized": True,
                "name": username,
                "faculty": "Engineering",
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "image_show": encoded_image_show,  # encoded_image değişkeni buraya yerleştirilmeli, show succeess screen.
                "image_detected": encoded_image_detected,  # latest_file, anlık foto
                "turnstile_id": turnstile_id
            }
        else:
            result = {
                "recognized": False,
                "name": None,
                "faculty": "None",
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "turnstile_id": -1
            }

        # Yerel klasöre JSON dosyasını kaydet
        local_json_path = 'D:\\TEDFaceRecognition\\data\\results_temp'
        local_json_path = os.path.join(local_json_path, f"{gateID}_result.json")
        with open(local_json_path, "w") as outfile:
            json.dump(result, outfile)

        # Yerel JSON dosyasını FTP sunucusuna yolla
        with open(local_json_path, "rb") as file:
            self.ftp.storbinary(f"STOR result.json", file)

        # Yerel JSON dosyasını sil
        #os.remove(local_json_path)

    def show_success_screen(self, image_location, name, faculty):
        current_time = time.time()

        # Check if it's the same person as the last detection
        if name == self.last_person_identity and current_time - self.last_success_time < 3:
            # Same person, apply cooldown
            time_remaining = int(3 - (current_time - self.last_success_time))
            # print(f"Same person detected. Cooldown: {time_remaining} seconds")
        else:
            # Different person or cooldown expired, show success screen
            self.last_person_identity = name
            self.success_screen_open = True
            if hasattr(self, 'success_screen') and self.success_screen:
                self.success_screen.destroy()

            self.success_screen = tk.Toplevel()
            self.success_screen.title("Successful login")
            self.success_screen.geometry("800x600")
            self.success_screen.configure(bg="#164B8B")

            circle_image_path = image_location
            circle_image = Image.open(circle_image_path)
            circle_image = circle_image.resize((170, 170))

            mask = Image.new("L", circle_image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, circle_image.size[0], circle_image.size[1]), fill=255)

            circle_image = ImageTk.PhotoImage(Image.composite(circle_image, Image.new("RGBA", circle_image.size), mask))

            logo_image = Image.open("D:\\TEDFaceRecognition\\logo.png")
            logo_image = logo_image.resize((100, 100))
            logo_image = ImageTk.PhotoImage(logo_image)

            frame = tk.Frame(self.success_screen, bg="#164B8B")
            frame.place(relx=0.5, rely=0.5, anchor="center")

            circle_label = tk.Label(frame, image=circle_image, bg="#164B8B")
            circle_label.grid(row=0, pady=10)

            text_label = tk.Label(frame, text=name, font=("Noyh R Light", 24), bg="#164B8B")
            text_label.grid(row=1, pady=10)

            bottom_text_label = tk.Label(frame, text="1234567", font=("Noyh R Light", 24), bg="#164B8B")
            bottom_text_label.grid(row=2, pady=10)

            faculty_label = tk.Label(frame, text=faculty, font=("Noyh R Light", 23), bg="#164B8B")
            faculty_label.grid(row=3, pady=10)

            logo_label = tk.Label(frame, image=logo_image, bg="#164B8B")
            logo_label.grid(row=4, pady=10)

            # Ensure the images are kept in memory
            circle_label.image = circle_image
            text_label.image = logo_image
            # Close the success screen if there is no new person in 3 seconds.
            self.success_screen.after(3000, self.hide_success_screen)
            self.last_success_time = current_time

            return self.success_screen
        #    self.success_screen.protocol("WM_DELETE_WINDOW", self.hide_success_screen)

    def hide_success_screen(self):
        # Hide the success screen without destroying it
        if self.success_screen_open and self.success_screen and self.success_screen.winfo_exists():
            self.success_screen.withdraw()
            self.success_screen_open = False

    def quit_app(self):

        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()


class CustomStack:
    def __init__(self):
        self.stack = deque()
        self.turnstiles = [1, 2, 3, 4]  # Mevcut turnikeler
        self.user_assigned_turnstiles = {}  # Kullanıcılara atanmış turnike bilgisi

    def push(self, name, current_time):
        assigned_turnstile_id = self.assign_turnstile(name)
        self.stack.append((name, current_time, assigned_turnstile_id))

    def assign_turnstile(self, name):
        if name in self.user_assigned_turnstiles:
            assigned_turnstile_id = self.user_assigned_turnstiles[name]
            if assigned_turnstile_id != -1:  # Daha önce atanmış bir turnike varsa ve -1 değilse
                return assigned_turnstile_id  # Önceden atanan turnikeyi geri döndür

        available_turnstiles = [t for t in self.turnstiles if t not in self.user_assigned_turnstiles.values()]

        if available_turnstiles:  # Eğer boş turnike varsa birini atayalım
            assigned_turnstile = min(available_turnstiles, key=self.turnstiles.index)
            self.user_assigned_turnstiles[name] = assigned_turnstile
            return assigned_turnstile
        else:
            print("Tüm turnikeler dolu!")
            return -1  # Tüm turnikeler doluysa -1 döndür


    def pop_expired(self, current_time, expire_time=10):
        expired_users = []
        i = 0
        while i < len(self.stack):
            if current_time - self.stack[i][1] >= expire_time:
                expired_user = self.stack[i]
                del self.stack[i]
                expired_users.append(expired_user)
            else:
                i += 1

        for user in expired_users:
            if user[2] != -1:  # -1'de olmayan bir turnike ise
                self.turnstiles.append(user[2])  # Turnikeyi serbest bırak
                self.user_assigned_turnstiles[user[0]] = -1  # Kullanıcının turnike atanmamış olarak işaretlenmesi

    def print_stack(self):
        for item in self.stack:
            print(f"Name: {item[0]}, Time: {item[1]}, Assigned Turnstile: {item[2]}")

    def get_last_element_name(self):
        if self.stack:
            return self.stack[-1][0]
        else:
            return None

    def get_assigned_turnstile_id(self, id_name):
        if id_name in self.user_assigned_turnstiles:
            assigned_turnstile_id = self.user_assigned_turnstiles[id_name]
            if assigned_turnstile_id != -1:
                return assigned_turnstile_id
        return -1  # -1 olmayan bir turnike atanmamışsa veya kullanıcı bulunamazsa None döndür

    def is_user_available(self, username):
        if username in self.user_assigned_turnstiles:
            turnstile_id = self.user_assigned_turnstiles[username]
            if turnstile_id != -1:
                return False  # Kullanıcı atanmış ve turnike_id'si -1 değilse
        return True  # Kullanıcı atanmamış veya turnike_id'si -1 ise


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    root.mainloop()
