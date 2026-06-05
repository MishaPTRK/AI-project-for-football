from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.player_dict = {}
        self.team_colors = {}
        self.kmeans = None

    def GetPlayerColor(self,frame, bbox):
        crop_frame = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        crop_img = crop_frame[0:int(crop_frame.shape[0]) // 2, :]
        image_2d = crop_img.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(image_2d)

        labals = kmeans.labels_
        clustered_image = labals.reshape(crop_img.shape[0], crop_img.shape[1])

        corner_clustered_image = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0],
                                  clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clustered_image), key=corner_clustered_image.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []

        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]

            player_color = self.GetPlayerColor(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans( n_clusters=2,init="k-means++",n_init=1)

        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def GetPlayerTeam(self, frame, player_bbox, player_id):
        if self.kmeans is None:
            raise RuntimeError("Call assign_team_color before GetPlayerTeam")

        if player_id in self.player_dict:
            return self.player_dict[player_id]

        player_color = self.GetPlayerColor(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id+=1

        self.player_dict[player_id] = team_id
        return team_id