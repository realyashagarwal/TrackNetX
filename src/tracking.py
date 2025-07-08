import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math
from scipy.optimize import linear_sum_assignment

@dataclass
class DetectionBox:
    """Represents a detection bounding box"""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> Tuple[int, int]:
        return (int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2))

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

@dataclass
class TrackedPlayer:
    track_id: int
    class_name: str
    current_box: DetectionBox
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    age: int = 0
    hits: int = 0
    hit_streak: int = 0
    time_since_update: int = 0
    color: Tuple[int, int, int] = field(default_factory=lambda: tuple(np.random.randint(0, 255, 3).tolist()))
    velocity: Tuple[float, float] = (0.0, 0.0)
    acceleration: Tuple[float, float] = (0.0, 0.0)
    last_seen_frame: Optional[np.ndarray] = None
    appearance_features: Optional[np.ndarray] = None

    def __post_init__(self):
        self.history.append(self.current_box.center)
        self.hits = 1
        self.hit_streak = 1

    def update(self, detection: DetectionBox):
        old_center = self.current_box.center
        new_center = detection.center
        prev_velocity = self.velocity
        if self.history:
            self.velocity = (
                new_center[0] - old_center[0],
                new_center[1] - old_center[1]
            )
            self.acceleration = (
                self.velocity[0] - prev_velocity[0],
                self.velocity[1] - prev_velocity[1]
            )
        self.current_box = detection
        self.history.append(detection.center)
        self.age += 1
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0

    def predict_next_position(self) -> Tuple[int, int]:
        if len(self.history) < 2:
            return self.current_box.center
        current_pos = self.current_box.center
        px = current_pos[0] + self.velocity[0] + 0.5 * self.acceleration[0]
        py = current_pos[1] + self.velocity[1] + 0.5 * self.acceleration[1]
        return (int(px), int(py))

    def get_predicted_box(self) -> DetectionBox:
        pc = self.predict_next_position()
        cw, ch = self.current_box.width // 2, self.current_box.height // 2
        return DetectionBox(
            x1=pc[0] - cw,
            y1=pc[1] - ch,
            x2=pc[0] + cw,
            y2=pc[1] + ch,
            confidence=self.current_box.confidence,
            class_id=self.current_box.class_id,
            class_name=self.current_box.class_name
        )

class PlayerTracker:
    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 100,
        iou_threshold: float = 0.1,
        confidence_threshold: float = 0.5
    ):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.next_id = 1
        self.tracked_players: Dict[int, TrackedPlayer] = {}
        self.lost_players: Dict[int, TrackedPlayer] = {}
        self.reidentification_threshold = 0.6
        self.weight_distance = 0.3
        self.weight_iou = 0.5
        self.weight_confidence = 0.1
        self.weight_size = 0.1
        self.jitter_tolerance = 20
        self.min_hit_streak_for_stable = 3
        self.stable_track_bonus = 0.8

    def calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])

    def calculate_iou(self, box1: DetectionBox, box2: DetectionBox) -> float:
        x1, y1 = max(box1.x1, box2.x1), max(box1.y1, box2.y1)
        x2, y2 = min(box1.x2, box2.x2), min(box1.y2, box2.y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        union = box1.area + box2.area - inter
        return inter / union if union > 0 else 0.0

    def calculate_size_similarity(self, box1: DetectionBox, box2: DetectionBox) -> float:
        if box1.area == 0 or box2.area == 0:
            return 0.0
        return min(box1.area, box2.area) / max(box1.area, box2.area)

    def calculate_comprehensive_cost(
        self, tracked_player: TrackedPlayer, detection: DetectionBox
    ) -> float:
        if detection.class_name != tracked_player.class_name or detection.confidence < self.confidence_threshold:
            return float('inf')
        pred_pos = tracked_player.predict_next_position()
        pred_box = tracked_player.get_predicted_box()
        dist = self.calculate_distance(pred_pos, detection.center)
        if dist > self.max_distance + self.jitter_tolerance:
            return float('inf')
        dist_cost = dist / (self.max_distance + self.jitter_tolerance)
        iou_val = max(
            self.calculate_iou(tracked_player.current_box, detection),
            self.calculate_iou(pred_box, detection)
        )
        eff_iou = (
            max(0.05, self.iou_threshold - 0.05)
            if dist < self.jitter_tolerance else self.iou_threshold
        )
        if iou_val < eff_iou:
            return float('inf')
        iou_cost = 1 - iou_val
        conf_cost = 1 - detection.confidence
        size_cost = 1 - self.calculate_size_similarity(tracked_player.current_box, detection)
        total = (
            self.weight_distance * dist_cost
            + self.weight_iou * iou_cost
            + self.weight_confidence * conf_cost
            + self.weight_size * size_cost
        )
        if tracked_player.hit_streak >= self.min_hit_streak_for_stable:
            total *= self.stable_track_bonus
        if dist < self.jitter_tolerance:
            total *= 0.5
        return total

    def associate_detections_hungarian(
        self, detections: List[DetectionBox], frame: np.ndarray
    ) -> Dict[int, int]:
        if not self.tracked_players or not detections:
            return {}
        ids = list(self.tracked_players.keys())
        cost_mat = np.full((len(ids), len(detections)), float('inf'))
        for i, tid in enumerate(ids):
            for j, det in enumerate(detections):
                cost_mat[i, j] = self.calculate_comprehensive_cost(self.tracked_players[tid], det)
        finite = np.where(cost_mat == float('inf'), 1e6, cost_mat)
        rows, cols = linear_sum_assignment(finite)
        assoc: Dict[int, int] = {}
        for r, c in zip(rows, cols):
            if cost_mat[r, c] != float('inf'):
                assoc[ids[r]] = c
        return assoc

    def extract_appearance_features(self, frame: np.ndarray, box: DetectionBox) -> np.ndarray:
        roi = frame[max(0, box.y1):min(frame.shape[0], box.y2),
                   max(0, box.x1):min(frame.shape[1], box.x2)]
        if roi.size == 0:
            return np.zeros(96)
        rsz = cv2.resize(roi, (64, 64))
        hist = []
        for ch in range(3):
            h = cv2.calcHist([rsz], [ch], None, [16], [0, 256])
            hist.append(cv2.normalize(h, h).flatten())
        h_mid = rsz.shape[0] // 2
        top, bot = rsz[:h_mid], rsz[h_mid:]
        htop = cv2.normalize(
            cv2.calcHist([top], [0,1,2], None, [8,8,8], [0,256,0,256,0,256]),
            None
        ).flatten()[:16]
        hbot = cv2.normalize(
            cv2.calcHist([bot], [0,1,2], None, [8,8,8], [0,256,0,256,0,256]),
            None
        ).flatten()[:16]
        return np.concatenate(hist + [htop, hbot])

    def compare_appearance(self, features1: np.ndarray, features2: np.ndarray) -> float:
        if features1.size == 0 or features2.size == 0:
            return 0.0
        corr = np.corrcoef(features1, features2)[0,1]
        corr = corr if not np.isnan(corr) else 0.0
        dot = np.dot(features1, features2)
        norm = np.linalg.norm(features1) * np.linalg.norm(features2)
        cos = dot / norm if norm > 0 else 0.0
        chi = sum(
            ((f1 - f2)**2) / (f1 + f2)
            for f1, f2 in zip(features1, features2)
            if f1 + f2 > 0
        )
        chi_sim = 1 / (1 + chi)
        return max(0.0, 0.4*corr + 0.4*cos + 0.2*chi_sim)

    def update(self, detections: List[DetectionBox], frame: np.ndarray) -> List[TrackedPlayer]:
        valid = [d for d in detections if d.confidence >= self.confidence_threshold]
        # no detections → age out
        if not valid:
            for pid in list(self.tracked_players):
                p = self.tracked_players[pid]
                p.time_since_update += 1
                p.hit_streak = 0
                if p.time_since_update > min(3, self.max_disappeared // 3):
                    p.last_seen_frame = frame.copy() if frame is not None else None
                    self.lost_players[pid] = p
                    del self.tracked_players[pid]
            for lid in list(self.lost_players):
                if self.lost_players[lid].time_since_update > self.max_disappeared:
                    del self.lost_players[lid]
            return list(self.tracked_players.values())
        
        assoc = self.associate_detections_hungarian(valid, frame)
        updated = set()

        # matched
        for tid, idx in assoc.items():
            self.tracked_players[tid].update(valid[idx])
            updated.add(tid)

        # unmatched → re-ID or new
        for i, det in enumerate(valid):
            if i not in assoc.values():
                best = (None, float('inf'))
                for lid, lp in self.lost_players.items():
                    if det.class_name == lp.class_name and lp.time_since_update <= self.max_disappeared:
                        cost = self.calculate_comprehensive_cost(lp, det)
                        if cost != float('inf'):
                            nf = self.extract_appearance_features(frame, det)
                            of = self.extract_appearance_features(frame, lp.current_box)
                            sim = self.compare_appearance(nf, of)
                            score = cost * (1 - sim)
                            if sim > self.reidentification_threshold and score < best[1]:
                                best = (lid, score)
                if best[0] is not None:
                    lp = self.lost_players.pop(best[0])
                    lp.update(det)
                    lp.time_since_update = 0
                    self.tracked_players[best[0]] = lp
                    updated.add(best[0])
                else:
                    new_id = self.next_id
                    tp = TrackedPlayer(new_id, det.class_name, det)
                    self.tracked_players[new_id] = tp
                    updated.add(new_id)
                    self.next_id += 1

        # cleanup not updated
        for tid, tp in list(self.tracked_players.items()):
            if tid not in updated:
                tp.time_since_update += 1
                tp.hit_streak = 0
                if tp.time_since_update > min(3, self.max_disappeared // 3):
                    tp.last_seen_frame = frame.copy() if frame is not None else None
                    self.lost_players[tid] = tp
                    del self.tracked_players[tid]
        # purge old lost
        for lid in list(self.lost_players):
            if self.lost_players[lid].time_since_update > self.max_disappeared:
                del self.lost_players[lid]

        return list(self.tracked_players.values())

    def draw_tracks(self, frame: np.ndarray, tracked_players: List[TrackedPlayer]) -> np.ndarray:
        out = frame.copy()
        for p in tracked_players:
            b = p.current_box; c = p.color
            th = max(1, int(b.confidence * 3))
            cv2.rectangle(out, (b.x1, b.y1), (b.x2, b.y2), c, th)
            pos = p.predict_next_position()
            cv2.circle(out, pos, 3, c, -1)
            lbl = f"ID:{p.track_id} {p.class_name} ({b.confidence:.2f})"
            sz = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(out, (b.x1, b.y1 - sz[1] - 10), (b.x1 + sz[0] + 5, b.y1), c, -1)
            cv2.putText(out, lbl, (b.x1 + 2, b.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            pts = list(p.history)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                line_col = tuple(int(col * alpha) for col in c)
                cv2.line(out, pts[i-1], pts[i], line_col, 2)
            if p.velocity != (0.0, 0.0):
                ce = b.center
                ve = (int(ce[0] + p.velocity[0] * 3), int(ce[1] + p.velocity[1] * 3))
                cv2.arrowedLine(out, ce, ve, c, 2, tipLength=0.3)
        return out

    def get_statistics(self) -> Dict:
        active = list(self.tracked_players.values())
        return {
            'active_tracks': len(active),
            'lost_tracks': len(self.lost_players),
            'total_tracks_created': self.next_id - 1,
            'track_ids': [p.track_id for p in active],
            'lost_track_ids': list(self.lost_players.keys()),
            'average_track_age': np.mean([p.age for p in active]) if active else 0,
            'average_hit_streak': np.mean([p.hit_streak for p in active]) if active else 0,
            'class_distribution': {cls: len([p for p in active if p.class_name == cls]) for cls in set(p.class_name for p in active)}
        }

    def validate_track_consistency(self, detections: List[DetectionBox]) -> Dict:
        info = {
            'id_switches': 0,
            'track_continuity': {},
            'association_quality': []
        }
        assoc = self.associate_detections_hungarian(detections, np.zeros((480,640,3), dtype=np.uint8))
        info['association_coverage'] = len(assoc) / len(detections) if detections else 1.0
        for tid, p in self.tracked_players.items():
            info['track_continuity'][tid] = {
                'hits': p.hits,
                'hit_streak': p.hit_streak,
                'age': p.age,
                'time_since_update': p.time_since_update
            }
        return info
