from sklearn.cluster import KMeans
import numpy as np
import cv2

class TeamAssigner:
    def __init__(self):
        self.player_dict = {}
        self.team_colors = {}
        self.referee_color = None
        self.kmeans = None

    def GetPlayerColor(self, frame, bbox):
        crop_frame = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        crop_img = crop_frame[0:int(crop_frame.shape[0]) // 2, :]
        cv2.imshow("Frame", crop_img)
        image_2d = crop_img.reshape(-1, 3)

        kmeans = KMeans(n_clusters=2, random_state=0).fit(image_2d)
        labels = kmeans.labels_
        clustered_image = labels.reshape(crop_img.shape[0], crop_img.shape[1])

        corners = [clustered_image[0, 0], clustered_image[0, -1],
                   clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corners), key=corners.count)
        player_cluster = 1 - non_player_cluster

        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.GetPlayerColor(frame, bbox)
            player_colors.append(player_color)

        kmeans3 = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=0)
        kmeans3.fit(player_colors)

        labels = kmeans3.labels_
        centers = kmeans3.cluster_centers_

        cluster_sizes = np.bincount(labels)
        referee_cluster_idx = np.argmin(cluster_sizes)
        team_cluster_idxs = [i for i in range(3) if i != referee_cluster_idx]

        self.referee_color = centers[referee_cluster_idx]

        team_colors = [player_colors[i] for i, label in enumerate(labels)
                       if label != referee_cluster_idx]

        kmeans2 = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0)
        kmeans2.fit(team_colors)
        self.kmeans = kmeans2

        self.team_colors[1] = kmeans2.cluster_centers_[0]
        self.team_colors[2] = kmeans2.cluster_centers_[1]

    def GetPlayerTeam(self, frame, player_bbox, player_id):
        if self.kmeans is None:
            raise RuntimeError("Call assign_team_color before GetPlayerTeam")

        if player_id in self.player_dict:
            return self.player_dict[player_id]

        player_color = self.GetPlayerColor(frame, player_bbox)

        if self.referee_color is not None:
            dist_to_referee = np.linalg.norm(player_color - self.referee_color)
            dist_to_team1 = np.linalg.norm(player_color - self.team_colors[1])
            dist_to_team2 = np.linalg.norm(player_color - self.team_colors[2])

            REFEREE_MARGIN = 1.5

            if dist_to_referee * REFEREE_MARGIN < dist_to_team1 and \
                    dist_to_referee * REFEREE_MARGIN < dist_to_team2:
                return 0

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1
        self.player_dict[player_id] = team_id
        return team_id