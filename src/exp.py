import customtkinter as CTK
from PIL import Image
from utils import PROJECT_DIR

# --- Farben & Konstante ---
BACKGROUND_COLOR = "#212121"
SIDEBAR_COLOR = "#181818"
SIDEBAR_HOVER_COLOR = "#303030"
SEARCHBAR_COLOR = "#303030"
SEARCHBAR_HOVER_COLOR = "#454545"
TEXT_COLOR = "#CCC7B9"
SIDEBAR_INITIAL_WIDTH = 90
SIDEBAR_EXPANDED_WIDTH = 300
BUTTON_INITAL_WIDTH = 40
BUTTON_REAL_WIDTH = 64


class FadeOutLabel(CTK.CTkLabel):
    """Custom label that fades its text and background colors."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields = ('text_color', 'fg_color')
        self.colors = {key: self.cget(key) for key in self.fields}
        try:
            base_color = self.master.cget('bg')
        except Exception:
            try:
                base_color = self.master.cget('fg_color')
            except Exception:
                base_color = BACKGROUND_COLOR
        self.colors['base'] = base_color
        self.configure(**{key: self.colors['base'] for key in self.fields})
        for key, color in self.colors.items():
            if isinstance(color, str):
                rgb_color = self.winfo_rgb(color)
                self.colors[key] = (rgb_color, rgb_color)
            else:
                self.colors[key] = self.winfo_rgb(color[0]), self.winfo_rgb(color[1])
        self.transition = 0
        self._update_color()

    def _get_curr(self, start, end):
        rgb_to_hex = lambda rgb: '#{:02x}{:02x}{:02x}'.format(*[int(val / 256) for val in rgb])
        return rgb_to_hex([int(start[i] + (end[i] - start[i]) * self.transition) for i in range(3)])

    def _update_color(self):
        color_config = {
            key: (self._get_curr(self.colors[key][0], self.colors['base'][0]),
                  self._get_curr(self.colors[key][1], self.colors['base'][1]))
            for key in self.fields
        }
        self.configure(**color_config)

    def fade_out(self, step=0.05):
        if self.transition < 1:
            self.transition += step
            self._update_color()
            self.after(50, lambda: self.fade_out(step))
        else:
            self.transition = 1
            self._update_color()

    def fade_in(self, step=0.05):
        if self.transition > 0:
            self.transition -= step
            self._update_color()
            self.after(50, lambda: self.fade_in(step))
        else:
            self.transition = 0
            self._update_color()


class Dashboard(CTK.CTk):  # Besser von CTK.CTk erben
    def __init__(self):
        super().__init__()
        self.configure(fg_color=BACKGROUND_COLOR)  # fg_color statt bg bei CTk
        CTK.set_appearance_mode("dark")

        # --- Layout-Struktur ---
        # Sidebar (links)
        self.sidebar_frame = CTK.CTkFrame(
            master=self,
            width=SIDEBAR_INITIAL_WIDTH,
            fg_color=SIDEBAR_COLOR,
            corner_radius=0
        )
        self.sidebar_frame.pack(side="left", fill="y", expand=False)
        # Verhindert, dass die Sidebar ihre Größe durch innere Widgets ändert
        self.sidebar_frame.pack_propagate(False)

        # Content-Bereich (füllt den Rest)
        self.content_frame = CTK.CTkFrame(
            master=self,
            fg_color=BACKGROUND_COLOR,
            corner_radius=0,
        )
        self.content_frame.pack(side="left", fill="both", expand=True)

        # --- Widgets in der Sidebar ---
        # Sidebar Toggle Button
        try:
            button_image_path = PROJECT_DIR / "data" / "images" / "sidebar_w.png"
            button_image = CTK.CTkImage(Image.open(button_image_path))
        except FileNotFoundError:
            print(f"Warnung: Bild nicht gefunden unter {button_image_path}. Verwende Text-Button.")
            button_image = None

        self.sidebar_button = CTK.CTkButton(
            master=self.sidebar_frame,
            text='☰' if button_image is None else '',  # Fallback-Text
            image=button_image,
            width=BUTTON_INITAL_WIDTH,
            height=BUTTON_INITAL_WIDTH,
            fg_color="transparent",  # Besser für runde Buttons
            hover_color=SIDEBAR_HOVER_COLOR,
            corner_radius=BUTTON_INITAL_WIDTH // 2,
            command=self.toggle_sidebar
        )
        self.sidebar_button.pack(pady=20, padx=(SIDEBAR_INITIAL_WIDTH - BUTTON_REAL_WIDTH) // 2)

        # --- Widgets im Content-Bereich ---
        # Input Frame (unten, wie im Beispiel)
        self.input_frame = CTK.CTkFrame(
            self.content_frame, fg_color=SEARCHBAR_COLOR, height=56, corner_radius=28
        )
        # Mit pack am unteren Rand verankern und horizontal füllen
        self.input_frame.pack(side="bottom", fill="x", padx=30, pady=20)

        self.gallery_button = CTK.CTkButton(
            self.input_frame, text="🖼️", width=40, height=40,
            fg_color="transparent", hover_color=SEARCHBAR_HOVER_COLOR,
            text_color="#8e9397", font=("Segoe UI Emoji", 18)
        )
        self.gallery_button.pack(side="left", padx=(10, 5), pady=8)

        self.input_entry = CTK.CTkEntry(
            self.input_frame, fg_color="transparent", border_width=0,  # transparent für nahtlosen Look
            text_color=TEXT_COLOR, font=("Noto", 20),
            placeholder_text="Was willst du heute wissen?"
        )
        self.input_entry.pack(side="left", fill="both", expand=True, pady=8)

        self.input_entry.bind("<KeyRelease>", self.on_entry_change)

        try:
            send_image_path = PROJECT_DIR / "data" / "images" / "arrow_w.png"
            send_image_path = CTK.CTkImage(Image.open(send_image_path))
        except FileNotFoundError:
            print(f"Warnung: Bild nicht gefunden unter {send_image_path}. Verwende Text-Button.")
            send_image_path = None

        self.send_button = CTK.CTkButton(
            self.input_frame, text="", image=send_image_path, width=40, height=40, corner_radius=20,
            fg_color="transparent", hover_color=SEARCHBAR_HOVER_COLOR,
            text_color=TEXT_COLOR, font=("Segoe UI", 18, "bold")
        )
        self.send_button.pack(side="right", padx=(5, 10), pady=8)

        # Greeting Text (zentriert im verbleibenden Platz)
        self.greeting_text = FadeOutLabel(
            master=self.content_frame,
            text='Hallo, Du',
            font=("Noto", 72),
            fg_color=BACKGROUND_COLOR,
            text_color=TEXT_COLOR
        )
        # Mit place relativ zentrieren
        self.greeting_text.place(relx=0.5, rely=0.38, anchor="center")

    def toggle_sidebar(self):
        """Erweitert oder kollabiert die Sidebar."""
        current_width = self.sidebar_frame.cget("width")
        if current_width < SIDEBAR_EXPANDED_WIDTH:
            self._animate_sidebar_width(SIDEBAR_EXPANDED_WIDTH)
        else:
            self._animate_sidebar_width(SIDEBAR_INITIAL_WIDTH)

    def _animate_sidebar_width(self, target_width):
        """Animiert die Breite der Sidebar zu einem Zielwert."""
        current_width = self.sidebar_frame.cget("width")
        if current_width != target_width:
            step = 25
            if current_width < target_width:
                new_width = min(current_width + step, target_width)
            else:
                new_width = max(current_width - step, target_width)
            self.sidebar_frame.configure(width=new_width)
            # Die Layout-Manager (pack) kümmern sich automatisch um die Anpassung.
            # Kein manuelles Update mehr nötig.
            self.after(10, lambda: self._animate_sidebar_width(target_width))

    def on_entry_change(self, event=None):
        content = self.input_entry.get()
        if content:
            self.greeting_text.fade_out()
        else:
            self.greeting_text.fade_in()


if __name__ == "__main__":
    app = Dashboard()
    app.state("zoomed")
    app.title("Dynamic Dashboard")
    app.mainloop()
