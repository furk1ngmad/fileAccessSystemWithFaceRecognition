import cv2
import face_recognition
import pickle
import tkinter as tk
from tkinter import messagebox
from threading import Thread

class TreeNode:
    def __init__(self, role, files=None, children=None):
        self.role = role
        self.files = files or []
        self.children = children or []

class RoleHierarchy:
    def __init__(self):
        self.root = TreeNode("CEO", files=["All_Files"], children=[
            TreeNode("Manager", files=["Manager_Level_Files"], children=[
                TreeNode("Employee", files=["Employee_Level_Files"])
            ])
        ])

    def get_files_for_role(self, role):
        return self._get_files_for_role(self.root, role)

    def _get_files_for_role(self, node, role):
        if node.role.lower() == role.lower():
            return node.files
        for child in node.children:
            result = self._get_files_for_role(child, role)
            if result is not None:
                return result
        return None

    def get_all_roles(self):
        return [self._get_all_roles(node) for node in [self.root]]

    def _get_all_roles(self, node):
        roles = [node.role]
        for child in node.children:
            roles.extend(self._get_all_roles(child))
        return roles

class FaceRecognition:
    def __init__(self):
        self.known_faces = {}
        self.face_data_file = 'face_data.pkl'
        self.role_hierarchy = RoleHierarchy()
        self.load_known_faces()
        self.video_capture = cv2.VideoCapture(0)
        self.recognizing = False
        self.recognizer_thread = Thread(target=self.recognize_loop, daemon=True)
        self.current_user_role = None

    def load_known_faces(self):
        try:
            with open(self.face_data_file, 'rb') as file:
                self.known_faces = pickle.load(file)
        except FileNotFoundError:
            pass

    def save_known_faces(self):
        with open(self.face_data_file, 'wb') as file:
            pickle.dump(self.known_faces, file)

    def get_face_encoding(self, face_image):
        face_encoding = face_recognition.face_encodings(face_image)
        if len(face_encoding) > 0:
            return face_encoding[0]
        else:
            return None

    def learn_face(self):
        _, frame = self.video_capture.read()
        face_encoding = self.get_face_encoding(frame)

        if face_encoding is not None:
            while True:
                role = input("Enter your role: ").strip()
                if self.role_hierarchy.get_files_for_role(role) is not None:
                    self.known_faces[role] = {"encoding": face_encoding, "files": self.role_hierarchy.get_files_for_role(role)}
                    self.save_known_faces()
                    print(f"You have been recognized as {role}.")
                    self.current_user_role = role
                    break
                else:
                    print("You should enter a valid role. Valid roles are:", ", ".join(self.role_hierarchy.get_all_roles()))
        else:
            print("No face found. Please try again.")

    def check_file_access(self, file_path):
        if self.current_user_role is not None:
            access_files = self.known_faces[self.current_user_role]["files"]
            print(f"Checking file access for {self.current_user_role}...")
            if any(file.lower() in file_path.lower() for file in access_files):
                print(f"You have access to {file_path}.")
            else:
                print(f"Access denied. You don't have permission to access {file_path}.")
        else:
            print("You need to learn your face first.")

    def recognize_face(self):
        self.recognizing = not self.recognizing

    def recognize_loop(self):
        while True:
            if self.recognizing:
                _, frame = self.video_capture.read()
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces([data["encoding"] for data in self.known_faces.values()], face_encoding)

                    if True in matches:
                        first_match_index = matches.index(True)
                        role = list(self.known_faces.keys())[first_match_index]
                    else:
                        role = "Unknown"

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, f"Role: {role}", (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

                cv2.imshow('Video', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.save_known_faces()
                    break

        self.video_capture.release()
        cv2.destroyAllWindows()

    def show_hierarchy(self):
        all_roles = self.role_hierarchy.get_all_roles()
        messagebox.showinfo("Role Hierarchy", "Hierarchy:\n" + "\n".join(map(str, all_roles)))

    def start_recognizer_thread(self):
        self.recognizer_thread.start()

    def run(self):
        self.start_recognizer_thread()

        while True:
            cmd = input("Enter 'l' to learn face, 'f' to toggle face recognition, 'h' to show hierarchy, 'c' to check file access, 'q' to quit: ").strip()
            if cmd == 'l':
                self.learn_face()
            elif cmd == 'f':
                self.recognize_face()
            elif cmd == 'h':
                self.show_hierarchy()
            elif cmd == 'c':
                file_path = input("Enter the file path to check access: ").strip()
                self.check_file_access(file_path)
            elif cmd == 'q':
                self.save_known_faces()
                break

if __name__ == "__main__":
    recognizer = FaceRecognition()
    recognizer.run()
