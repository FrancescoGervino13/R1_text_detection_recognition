import numpy as np
import re
import math
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
from Levenshtein import distance as levenshtein_distance
from sklearn.metrics.pairwise import cosine_similarity
from openai import AzureOpenAI
from utils.utils import load_config

import time

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


def count_words(s):
    return len(s.split())

def sigmoid(x, k=1, x0=2.5):
    return 1 / (1 + math.exp(k * (x - x0)))

def euclidean_distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

def compute_obb_corners(centre, width, height, thickness, x, y, n) :

    n = np.cross(x, y)
    n = n / np.linalg.norm(n)  # Normalize the normal vector

    w, h, t = width / 2, height / 2, thickness / 2

    corners = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                offset = sx * w * x + sy * h * y + sz * t * n
                corners.append(centre + offset)
    return np.array(corners)

class DemoNode(Node):
    def __init__(self):
        super().__init__('demo_node')
        start = time.time()
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cuda')
        self.texts = []
        self.data = []
        self.declare_parameters(
            namespace = '',
            parameters = [
                ('config_path', '/home/user1/config.env'),      # Config Path for AZURE chatGPT keys
                ('chatgpt_model_name', 'hsp-Vocalinteraction_gpt4o'),
                ('img_topic_name', 'image_and_bboxes'),         # Input of type ImageBoundingBoxes
                ('result_topic_name', 'aligned_texts_clouds')   # Output of type AlignedTextsClouds
            ])
        openai_config = load_config(self.get_parameter('config_path').value)
        self.client = AzureOpenAI(
            azure_endpoint = f"{openai_config['AZURE_ENDPOINT']}",
            api_key = openai_config['AZURE_API_KEY'],
            api_version = "2024-10-21"
            )
        self.chatgpt_model_name = self.get_parameter('chatgpt_model_name').value
        end = time.time()
        print(f"Model loaded in {end - start:.2f} seconds")

        self.text_publisher = self.create_publisher(Marker, 'text_markers', 5)
        self.bbox_publisher = self.create_publisher(Marker, 'obb_lines', 5)
        self.final_bbox_publisher = self.create_publisher(Marker, 'bboxes', 5)
        self.final_text_publisher = self.create_publisher(Marker, 'texts', 5)
        self.box_id = 0
        self.final_box_id = 1000
        self.colour_selector = 0
        self.final_colour_selector = 0

        start = time.time()

        print("Reading recognized texts from file...")
        with open("/home/fgervino-iit.local/visual-language-navigation/mmocr_ros/text_rec/mmocr_ros/recognized_texts.txt", "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip().replace('+', '')
                parts = re.findall(r'\[([^\]]+)\]', line)

                if len(parts) != 6:
                    print(f"Line {line_num} skipped: expected 6 bracketed parts but found {len(parts)}")
                    continue

                name = parts[0].strip().strip('"')
                self.texts.append(name)

                try:
                    parts[1] = parts[1].replace(',', '')
                    parts[2] = parts[2].replace(',', '')
                    parts[3] = parts[3].replace(',', '')
                    parts[4] = parts[4].replace(',', '')
                    parts[5] = parts[5].replace(',', '')
                    point3D = list(map(float, parts[1].strip().split()))
                    dim2D = list(map(float, parts[2].strip().split()))
                    vec1 = list(map(float, parts[3].strip().split()))
                    vec2 = list(map(float, parts[4].strip().split()))
                    vec3 = list(map(float, parts[5].strip().split()))

                    '''corners = compute_obb_corners(point3D, dim2D[0], dim2D[1], 0.01, np.array(vec1), np.array(vec2), np.array(vec3))
                    self.publish_obb_lines(False, corners)
                    self.publish_text_marker(False, name, point3D)
                    self.box_id += 1
                    time.sleep(0.05)'''
                
                except ValueError as e:
                    print(f"Line {line_num} skipped: cannot convert to float - {e}")
                    continue

                self.data.append([point3D, dim2D, vec1, vec2, vec3])

        self.N = len(self.texts)
    
        print("Encoding SBERT embeddings...")
        self.embeddings = self.model.encode(self.texts, convert_to_tensor=True, device='cuda')
        print("Computing SBERT cosine similarity matrix...")
        self.sbert_sim_matrix = cosine_similarity(self.embeddings.cpu().numpy())
        print("Computing Levenshtein similarity matrix...")
        self.lev_sim_matrix = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i, self.N):
                max_len = max(len(self.texts[i]), len(self.texts[j]))
                sim = 1 - levenshtein_distance(self.texts[i], self.texts[j]) / max_len if max_len > 0 else 1
                self.lev_sim_matrix[i, j] = self.lev_sim_matrix[j, i] = sim

        print("Computing custom distance matrix...")
        self.distance_matrix = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i + 1, self.N):
                dist = self.compute_distance(i, j)
                self.distance_matrix[i, j] = self.distance_matrix[j, i] = dist

        print("Clustering...")
        clustering = DBSCAN(eps=0.01, min_samples=5, metric='precomputed').fit(self.distance_matrix)
        labels = clustering.labels_
        print(f"Clustering completed in {time.time() - start:.2f} seconds")

        def semantic_similarity(text1, text2):
            words = (count_words(text1) + count_words(text2)) / 2
            sw = sigmoid(words) # Semantic weight based on the number of words

            lev = levenshtein_distance(text1, text2)
            sbert = cosine_similarity(
                [self.model.encode(text1, convert_to_tensor=True, device='cuda').cpu().numpy()],
                [self.model.encode(text2, convert_to_tensor=True, device='cuda').cpu().numpy()]
            )[0][0]

            semantic_similarity = sw * lev + (1 - sw) * sbert
            return semantic_similarity

        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(idx)

        cluster_texts = []
        cluster_centres = []
        cluster_dimensions = []
        cluster_x_axes = []
        cluster_y_axes = []
        cluster_z_axes = []
        for label, indices in clusters.items():
            print(f"Cluster {label}: {[i + 1 for i in indices]}")
            if label != -1 :
                '''if label == 5 or label == 6 :
                    cluster_texts.append("SCALA EST EAST STAIRCASE")
                elif label == 14 :
                    cluster_texts.append("TAKE A BOOK AND ENJOY READINC IT! BUT DONT FORGETTO FILL THE BOOKING FORM")
                else :'''
                cluster_texts.append(self.real_text([self.texts[i] for i in indices]))
                print(f"Cluster {label} text: {cluster_texts[-1]}")

                cluster_weight = 0.0
                for i in indices:
                    w = semantic_similarity(cluster_texts[-1], self.texts[i])
                    if i == indices[0] :
                        cluster_centre = np.array(self.data[i][0])
                        cluster_dimension = np.array(self.data[i][1])
                        cluster_x_axis = np.array(self.data[i][2])
                        cluster_y_axis = np.array(self.data[i][3])
                        cluster_z_axis = np.array(self.data[i][4])
                        cluster_weight = w
                    else :
                        a = w / (cluster_weight + w)
                        b = cluster_weight / (cluster_weight + w)
                        cluster_centre = a * np.array(self.data[i][0]) + b * cluster_centre
                        cluster_dimension = a * np.array(self.data[i][1]) + b * cluster_dimension
                        cluster_x_axis = a * np.array(self.data[i][2]) + b * cluster_x_axis
                        cluster_y_axis = a * np.array(self.data[i][3]) + b * cluster_y_axis
                        cluster_z_axis = a * np.array(self.data[i][4]) + b * cluster_z_axis
                        cluster_weight += w
                    
                cluster_centres.append(cluster_centre)
                cluster_dimensions.append(cluster_dimension)
                cluster_x_axes.append(cluster_x_axis)
                cluster_y_axes.append(cluster_y_axis)
                cluster_z_axes.append(cluster_z_axis)
                
                corners = compute_obb_corners(cluster_centres[-1], cluster_dimensions[-1][0], cluster_dimensions[-1][1], 0.01, 
                                            np.array(cluster_x_axes[-1]), 
                                            np.array(cluster_y_axes[-1]), 
                                            np.array(cluster_z_axes[-1]))
                self.publish_obb_lines(True, corners)
                self.publish_text_marker(True, cluster_texts[-1], cluster_centres[-1])
                self.final_box_id += 1
                time.sleep(0.05)

        piero = 0

    def publish_obb_lines(self, final, corners, frame_id="map"):
        COLOR_LIST = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 0.5, 0.0),
        ]

        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rclpy.clock.Clock().now().to_msg()
        marker.id = self.box_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        if final :
            r, g, b = COLOR_LIST[self.final_colour_selector % len(COLOR_LIST)]
            marker.ns = "merged_obb_lines"
            self.final_colour_selector += 1
        else :
            r, g, b = COLOR_LIST[self.colour_selector % len(COLOR_LIST)]
            marker.ns = "obb_lines"
            self.colour_selector += 1
        
        marker.scale.x = 0.01
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0

        marker.points = []
        edges = [
            (0,1), (1,3), (3,2), (2,0),
            (4,5), (5,7), (7,6), (6,4),
            (0,4), (1,5), (2,6), (3,7)
        ]

        for i, j in edges:
            p1 = Point(x=float(corners[i][0]), y=float(corners[i][1]), z=float(corners[i][2]))
            p2 = Point(x=float(corners[j][0]), y=float(corners[j][1]), z=float(corners[j][2]))
            marker.points.extend([p1, p2])

        marker.lifetime.sec = 0

        if final :
            marker.id = self.final_box_id
            self.final_bbox_publisher.publish(marker)
        else :
            self.bbox_publisher.publish(marker)

    def publish_text_marker(self, final, text, position, frame_id="map", scale=0.1, color=(1.0, 1.0, 1.0)):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rclpy.clock.Clock().now().to_msg()
        marker.id = self.box_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position = Point(
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2] + 0.1)  # Offset above box for visibility
        )

        marker.scale.z = scale  # Only Z is used for text size

        r, g, b = color
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0

        marker.text = text
        marker.lifetime.sec = 0

        if final :
            marker.id = self.final_box_id
            marker.ns = "merged_texts"
            self.final_text_publisher.publish(marker)
        else :
            marker.ns = "box_text"
            self.text_publisher.publish(marker)
    
    def compute_distance(self, i, j):
        text_x = self.texts[i]
        text_y = self.texts[j]
        words = (count_words(text_x) + count_words(text_y)) / 2

        # Geometry
        cx = np.array(self.data[i][0])
        cy = np.array(self.data[j][0])
        dimx = np.array(self.data[i][1])
        dimy = np.array(self.data[j][1])
        avg_dimx = 0.75 * max(dimx[0], dimx[1]) + 0.25 * min(dimx[0], dimx[1])
        avg_dimy = 0.75 * max(dimy[0], dimy[1]) + 0.25 * min(dimy[0], dimy[1])
        thresh = (avg_dimx + avg_dimy) / 2

        nx = np.array(self.data[i][4])
        ny = np.array(self.data[j][4])
        gw = 1.5 - 0.5 * np.dot(nx, ny) # Geometric weight mesaured by cosine similarity
        sw = sigmoid(words) # Semantic weight based on the number of words

        lev = self.lev_sim_matrix[i, j]
        sbert = self.sbert_sim_matrix[i, j]

        geometric_dist = euclidean_distance(cx, cy) * gw
        semantic_similarity = sw * lev + (1 - sw) * sbert
        w = 1 - 1.5 * semantic_similarity # Semantic weight that diminishes or enhances the geometric distance

        distance = geometric_dist + w * thresh
        return max(distance - thresh + 0.01, 0.0)
    
    def is_text_safe(self, text):
        try:
            response = self.client.chat.completions.create(
                model=self.chatgpt_model_name,
                messages=[
                    {"role": "system", "content": "Repeat this text back."},
                    {"role": "user", "content": text}
                ]
            )
            _ = response.choices[0].message.content.strip()
            return True
        except Exception as e:
            print(f"Filtered out suspicious text: '{text}' — {e}")
            return False

    def real_text(self, texts):
        """
        Given a list of texts, use the language model to identify which is probably the text of the image.
        
        Args:
            texts (list): List of strings
        
        Returns:
            str: Image's text as identified by the language model
        """

        '''
        # Step 1: Remove any suspicious or filtered texts
        safe_texts = [t for t in texts if self.is_text_safe(t)]

        if not safe_texts:
            print("No safe texts to process.")
            return None
        '''

        prompt = """
            You are given a list of texts. This list contains words of texts detected by OCR in a image in an indoor space.
            Your task is to identify the real text of the image, which is the most likely to be the text present in the image.
            The texts may contain noise, such as artifacts from the OCR process or irrelevant words.
            Please analyze the texts and return the one that you believe is the most accurate representation of the image's text.
            If you are unsure, you can return the most relevant text based on your understanding of the context.
            If you find that all texts are equally valid, you can return any one of them.
            Return only the text, without any additional explanation or formatting.
            """

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"List of texts: {texts}"}
        ]

        response = self.client.chat.completions.create(
            model="hsp-Vocalinteraction_gpt4o",  
            messages=messages
        )

        try:
            reply = response.choices[0].message.content.strip()
            return reply
        except Exception as e:
            if "LOCALE QUADRI ELETTRICI" in texts:
                return "LOCALE QUADRI ELETTRICI"
            else: 
                return "ChatGPT fa schifo"

def main(args=None):
    rclpy.init(args=args)
    node = DemoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()