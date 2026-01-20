import cv2
import numpy as np
import os
import pandas as pd
import torch

# --- KONFIGURATION ---
your_path = "C:/Users/black/Documents/Python/Python/Liverdata"
processed_path = os.path.join(your_path, "Processed")
training_pics_path = os.path.join(your_path, "Training pics")

IMAGE_NAMES = ["M1", "M2", "M3"] 
DISPLAY_SIZE = 1200  # Storleken på fönstret på din skärm

COLORS = {
    'central': (0, 0, 255),  # Röd (BGR format för OpenCV)
    'portal': (255, 0, 0)    # Blå
}
COLOR_NAMES = ['central', 'portal']

class AnnotationTool:
    def __init__(self, image_names):
        self.image_names = image_names
        self.current_idx = 0
        self.current_structure = 'central'
        
        self.zoom_level = 1.0
        self.offset = [0, 0]
        self.is_panning = False
        self.pan_start = (0, 0)
        
        self.image = None
        self.img_h = 0
        self.img_w = 0
        self.annotations = {'central': [], 'portal': []}
        
        if not os.path.exists(training_pics_path):
            os.makedirs(training_pics_path)
            
        self.load_image()
    
    def load_image(self):
        if self.current_idx >= len(self.image_names): return
        img_name = self.image_names[self.current_idx]
        
        high_res_path = None
        for ext in ['.jpg', '.jpeg', '.tif', '.png']:
            p = os.path.join(training_pics_path, f"{img_name}{ext}")
            if os.path.exists(p):
                high_res_path = p
                break
        
        if high_res_path:
            print(f"Loading High-Res Image: {high_res_path}")
            # Läs in bilden
            self.image = cv2.imread(high_res_path)
        else:
            print(f"❌ Kunde inte hitta bilden: {img_name} i {training_pics_path}")
            self.image = None
            return

        if self.image is None:
            print(f"❌ Bildfilen är tom eller trasig: {img_name}")
            return

        self.img_h, self.img_w = self.image.shape[:2]
        self.zoom_level = 1.0
        self.offset = [0, 0] # Reset view
        self.annotations = {'central': [], 'portal': []}
        
        print(f"✅ Loaded {img_name} | Resolution: {self.img_w}x{self.img_h}")
        print("Högprecisions-läge aktiverat. Klick sparas som exakta pixlar.")

    def mouse_callback(self, event, x, y, flags, param):
        if self.image is None: return
        
        # Räkna ut var vi är i den STORA bilden
        real_x = int(x / self.zoom_level + self.offset[0])
        real_y = int(y / self.zoom_level + self.offset[1])
        
        # Säkerställ att vi inte klickar utanför bilden
        real_x = max(0, min(real_x, self.img_w - 1))
        real_y = max(0, min(real_y, self.img_h - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            # SPARA EXACTA PIXLAR (Ingen nedskalning!)
            self.annotations[self.current_structure].append((real_x, real_y))
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Ångra senaste klicket nära musen (inom 50 pixlar) eller ta sista
            if self.annotations[self.current_structure]:
                # Smart undo: Ta bort den punkt som är närmast klicket
                pts = self.annotations[self.current_structure]
                dists = [np.sqrt((px-real_x)**2 + (py-real_y)**2) for px, py in pts]
                min_dist_idx = np.argmin(dists)
                
                if dists[min_dist_idx] < 100 / self.zoom_level: # Om vi klickar nära en punkt
                    pts.pop(min_dist_idx)
                else:
                    pts.pop() # Annars ta bort sista

        elif event == cv2.EVENT_MBUTTONDOWN:
            self.is_panning = True
            self.pan_start = (x, y)

        elif event == cv2.EVENT_MBUTTONUP:
            self.is_panning = False

        elif event == cv2.EVENT_MOUSEMOVE and self.is_panning:
            self.offset[0] += int((self.pan_start[0] - x) / self.zoom_level)
            self.offset[1] += int((self.pan_start[1] - y) / self.zoom_level)
            self.pan_start = (x, y)

        elif event == cv2.EVENT_MOUSEWHEEL:
            # Zooma mot musens position
            old_zoom = self.zoom_level
            if flags > 0: self.zoom_level *= 1.25
            else: self.zoom_level = max(0.1, self.zoom_level / 1.25)
            
            # Justera offset så vi zoomar in där musen pekar
            scale = self.zoom_level / old_zoom
            self.offset[0] = int(real_x - (x / self.zoom_level))
            self.offset[1] = int(real_y - (y / self.zoom_level))


    def run(self):
        win = 'Annotation Tool'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, DISPLAY_SIZE, DISPLAY_SIZE)
        cv2.setMouseCallback(win, self.mouse_callback)
        
        while self.current_idx < len(self.image_names):
            if self.image is None:
                self.current_idx += 1
                if self.current_idx < len(self.image_names): self.load_image()
                continue

            # --- OPTIMERING ---
            # Vi ritar bara på den delen av bilden vi faktiskt ser (View)
            # Istället för att rita på hela 10GB-bilden varje frame.
            
            # 1. Räkna ut viewport
            h_view = int(DISPLAY_SIZE / self.zoom_level)
            w_view = int(DISPLAY_SIZE / self.zoom_level)
            
            # Clamp offset
            self.offset[0] = max(0, min(self.offset[0], self.img_w - w_view))
            self.offset[1] = max(0, min(self.offset[1], self.img_h - h_view))
            
            x_start, y_start = int(self.offset[0]), int(self.offset[1])
            x_end, y_end = x_start + w_view, y_start + h_view
            
            # Se till att vi inte går utanför bildens kanter
            x_end = min(x_end, self.img_w)
            y_end = min(y_end, self.img_h)
            
            # Klipp ut det vi ser (Detta går snabbt)
            view_crop = self.image[y_start:y_end, x_start:x_end].copy()
            
            # Om vi zoomar ut för långt (bilden är mindre än fönstret)
            if view_crop.shape[0] == 0 or view_crop.shape[1] == 0:
                 # Fallback om man zoomar ut galet mycket
                 blank = np.zeros((DISPLAY_SIZE, DISPLAY_SIZE, 3), dtype=np.uint8)
                 cv2.putText(blank, "Zoomed out too far", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
                 cv2.imshow(win, blank)
                 if cv2.waitKey(10) & 0xFF == ord('q'): break
                 continue

            # 2. Rita annoteringar BARA på denna lilla crop (Mycket snabbare!)
            # Vi måste justera koordinaterna relativt till croppen
            
            # Punktstorlek som inte blir osynlig när man zoomar ut
            radius = int(50 / self.zoom_level) 
            radius = max(3, min(radius, 100)) # Minst 3px, max 100px
            thickness = -1 # Fylld cirkel
            
            for s_type, pts in self.annotations.items():
                color = COLORS[s_type]
                for (real_x, real_y) in pts:
                    # Kolla om punkten är inom vår synliga vy
                    if x_start <= real_x < x_end and y_start <= real_y < y_end:
                        # Rita relativt till view_crop
                        draw_x = int(real_x - x_start)
                        draw_y = int(real_y - y_start)
                        
                        cv2.circle(view_crop, (draw_x, draw_y), radius, color, thickness)
                        # Vit kant för synlighet
                        cv2.circle(view_crop, (draw_x, draw_y), radius+2, (255,255,255), 2)

            # 3. Skala upp/ner croppen till fönsterstorlek
            final_view = cv2.resize(view_crop, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)

            # UI Text
            info = f"{self.image_names[self.current_idx]} | {self.current_structure.upper()} | Zoom:{self.zoom_level:.2f}x"
            cv2.putText(final_view, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(final_view, f"Points: {len(self.annotations[self.current_structure])}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow(win, final_view)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('1'): self.current_structure = 'central'
            elif key == ord('2'): self.current_structure = 'portal'
            elif key == ord('s'):
                self.save_annotations()
                self.current_idx += 1
                if self.current_idx < len(self.image_names): self.load_image()
            elif key == ord('q'): break
            
            # WASD Panning
            pan_step = int(w_view * 0.1)
            if key == ord('i'): self.offset[1] -= pan_step
            elif key == ord('k'): self.offset[1] += pan_step
            elif key == ord('j'): self.offset[0] -= pan_step
            elif key == ord('l'): self.offset[0] += pan_step

        cv2.destroyAllWindows()
        print(f"Annotation complete!")

    def save_annotations(self):
        img_name = self.image_names[self.current_idx]
        
        for structure_type in COLOR_NAMES:
            if self.annotations[structure_type]:
                # Spara som DataFrame
                df = pd.DataFrame(self.annotations[structure_type], columns=['x', 'y'])
                
                # Spara
                output_path = os.path.join(training_pics_path, f"{img_name}_{structure_type}.csv")
                df.to_csv(output_path, index=False, header=False)
                print(f"  Saved {len(self.annotations[structure_type])} {structure_type} points to {output_path}")
            else:
                print(f"  No {structure_type} points marked")
        
        print(f"✓ Annotations saved for {img_name}\n")

if __name__ == "__main__":
    print(f"Annotating: {IMAGE_NAMES}")
    print("Controls: 1=Central (RED), 2=Portal (BLUE)")
    print("Mouse: Left=Add, Right=Undo, Scroll=Zoom, Middle=Pan")
    print("Keys: WASD=Pan, S=Save & Next, Q=Quit")
    
    tool = AnnotationTool(IMAGE_NAMES)
    tool.run()