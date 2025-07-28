# -*- coding: utf-8 -*-
"""
ASR Model Training GUI with Real-Time Logging, Parameter Control, and Early Stopping.
Framework: PyQt6 + TensorFlow/Keras

This is the final, complete, and self-contained script with all features,
including a themed UI, embedded TensorBoard, save/load functionality, and an
expanded menu with external links.
"""
import sys
import logging
import re
import os
import json
from pathlib import Path
from glob import glob
from datetime import datetime
import traceback
import subprocess

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QPushButton,
    QLineEdit,
    QTextEdit,
    QSplitter,
    QLabel,
    QMessageBox,
    QFileDialog,
    QProgressBar,
    QStatusBar,
    QTabWidget,
)
from PyQt6.QtCore import QObject, QThread, pyqtSignal, Qt, QUrl, QSettings
from PyQt6.QtGui import QAction, QIntValidator, QDoubleValidator, QDesktopServices
from PyQt6.QtWebEngineWidgets import QWebEngineView

# --- ML Imports ---
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Suppress TensorFlow INFO messages ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.get_logger().setLevel("ERROR")


# --- PART 1: PYQT6 THREADING AND COMMUNICATION SETUP ---
class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    message = pyqtSignal(str)
    tensorboard_path = pyqtSignal(str)
    progress = pyqtSignal(int)


class SignalLogHandler(logging.Handler):
    def __init__(self, signals, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signals = signals

    def emit(self, record):
        self.signals.message.emit(self.format(record))


class StopTrainingCallback(keras.callbacks.Callback):
    def __init__(self, worker):
        super().__init__()
        self.worker = worker

    def on_batch_end(self, batch, logs=None):
        if not self.worker._is_running:
            self.model.stop_training = True


class GUIDisplayOutputs(keras.callbacks.Callback):
    def __init__(self, batch, vectorizer, signals, beam_width=3, log_every=5):
        super().__init__()
        self.batch = batch
        self.vectorizer = vectorizer
        self.signals = signals
        self.beam_width = beam_width
        self.log_every = log_every
        self.start_idx = self.vectorizer.char_to_idx["<"]
        self.end_idx = self.vectorizer.char_to_idx[">"]

    def decode(self, indices):
        return "".join(
            [
                self.vectorizer.idx_to_char.get(i, "?")
                for i in indices
                if i not in (self.start_idx, self.end_idx, 0)
            ]
        )

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        lr = self.model.optimizer.learning_rate(self.model.optimizer.iterations)
        self.signals.message.emit(
            f"Epoch {epoch + 1}: val_loss: {logs.get('val_loss', 0):.4f}, Learning Rate: {lr:.8f}"
        )
        if (epoch + 1) % self.log_every == 0:
            self.signals.message.emit(
                f"\n----- Predictions (Beam Width: {self.beam_width}) at Epoch {epoch + 1} -----"
            )
            preds = self.model.beam_search_decode(
                self.batch["source"],
                self.start_idx,
                self.end_idx,
                beam_width=self.beam_width,
            ).numpy()
            for i in range(min(self.batch["source"].shape[0], 4)):
                target_text = self.decode(self.batch["target"].numpy()[i])
                pred_text = self.decode(preds[i])
                self.signals.message.emit(f"Target      : {target_text}")
                self.signals.message.emit(f"Prediction  : {pred_text}\n")
            self.signals.message.emit(
                "----------------------------------------------------------"
            )


class ProgressCallback(keras.callbacks.Callback):
    def __init__(self, signals):
        super().__init__()
        self.signals = signals

    def on_epoch_begin(self, epoch, logs=None):
        self.signals.progress.emit(0)

    def on_train_batch_end(self, batch, logs=None):
        if self.params["steps"]:
            self.signals.progress.emit(int(100 * (batch + 1) / self.params["steps"]))

    def on_epoch_end(self, epoch, logs=None):
        self.signals.progress.emit(100)


# --- PART 2: FULL ASR MODEL AND DATA PROCESSING CODE ---
class HParams:
    SAMPLE_RATE = 8000
    FFT_LENGTH = 256
    FRAME_LENGTH = 200
    FRAME_STEP = 80
    NUM_MEL_BINS = 80
    MAX_AUDIO_SECS = 15
    AUDIO_LEN = int((MAX_AUDIO_SECS * SAMPLE_RATE - FRAME_LENGTH) / FRAME_STEP + 1)
    MAX_TARGET_LEN = 200
    NUM_HID = 256
    NUM_HEAD = 4
    FF_DIM = 1024
    NUM_LAYERS_ENC = 4
    NUM_LAYERS_DEC = 4
    DROPOUT_RATE = 0.2
    VALIDATION_SPLIT = 0.99
    BATCH_SIZE = 32
    EPOCHS = 200
    INIT_LR = 1e-6
    WARMUP_LR = 1e-4
    FINAL_LR = 1e-6
    WARMUP_EPOCHS = 10
    DECAY_EPOCHS = 90
    BEAM_WIDTH = 3


def normalize_text(name):
    return os.path.splitext(os.path.basename(name.strip()))[0]


def get_data(wavs_path, transcript_path, maxlen):
    wavs = glob(str(wavs_path / "*.wav"))
    id_to_text = {}
    tuple_pattern = re.compile(r'\("([^"]+)",\s*(.+)\s*\)')
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            match = tuple_pattern.match(line.strip())
            if match:
                id_to_text[normalize_text(match.group(1))] = match.group(2).strip("'\"")
    data, unmatched = [], []
    for w in wavs:
        file_id = normalize_text(w)
        if file_id in id_to_text and len(id_to_text[file_id]) < maxlen:
            data.append({"audio": str(w), "text": id_to_text[file_id]})
        else:
            unmatched.append(file_id)
    return data, unmatched


class VectorizeChar:
    def __init__(self, max_len):
        self.vocab = ["-", "#", "<", ">"] + list("abcdefghijklmnopqrstuvwxyz .,?")
        self.max_len, self.char_to_idx = max_len, {
            ch: i for i, ch in enumerate(self.vocab)
        }
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}

    def __call__(self, text):
        text = "<" + text.lower()[: self.max_len - 2] + ">"
        return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * (
            self.max_len - len(text)
        )

    def get_vocabulary(self):
        return self.vocab


def path_to_audio(path, hparams):
    try:
        audio, _ = tf.audio.decode_wav(tf.io.read_file(path), 1)
        audio = tf.squeeze(audio, -1)
        stfts = tf.signal.stft(
            audio,
            frame_length=hparams.FRAME_LENGTH,
            frame_step=hparams.FRAME_STEP,
            fft_length=hparams.FFT_LENGTH,
        )
        spectrograms = tf.abs(stfts)
        num_spectrogram_bins = stfts.shape[-1]
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            hparams.NUM_MEL_BINS, num_spectrogram_bins, hparams.SAMPLE_RATE, 20, 4000
        )
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(
            spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])
        )
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        means = tf.reduce_mean(log_mel_spectrograms, 1, keepdims=True)
        stddevs = tf.math.reduce_std(log_mel_spectrograms, 1, keepdims=True)
        x = (log_mel_spectrograms - means) / (stddevs + 1e-6)
        x = x[: hparams.AUDIO_LEN, :]
        pad_len = hparams.AUDIO_LEN - tf.shape(x)[0]
        return tf.pad(x, [[0, pad_len], [0, 0]], "CONSTANT")
    except Exception:
        return tf.zeros((hparams.AUDIO_LEN, hparams.NUM_MEL_BINS))


def create_tf_dataset(data, vectorizer, hparams, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(
        {
            "audio": [s["audio"] for s in data],
            "text": [vectorizer(s["text"]) for s in data],
        }
    )
    ds = ds.map(
        lambda x: {"source": path_to_audio(x["audio"], hparams), "target": x["text"]},
        tf.data.AUTOTUNE,
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab, maxlen, num_hid):
        super().__init__()
        self.emb = keras.layers.Embedding(num_vocab, num_hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        return x + self.pos_emb(positions)


class SpeechFeatureEmbedding(layers.Layer):
    def __init__(self, num_hid):
        super().__init__()
        self.conv1 = layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv2 = layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv3 = layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )

    def call(self, x):
        return self.conv3(self.conv2(self.conv1(x)))


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(feed_forward_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1, self.layernorm2 = layers.LayerNormalization(
            epsilon=1e-6
        ), layers.LayerNormalization(epsilon=1e-6)
        self.dropout1, self.dropout2 = layers.Dropout(rate), layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs, training=training)
        out1 = self.layernorm1(inputs + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layernorm1, self.layernorm2, self.layernorm3 = [
            layers.LayerNormalization(epsilon=1e-6) for _ in range(3)
        ]
        self.self_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_dropout, self.enc_dropout, self.ffn_dropout = [
            layers.Dropout(dropout_rate) for _ in range(3)
        ]
        self.ffn = keras.Sequential(
            [layers.Dense(feed_forward_dim, activation="relu"), layers.Dense(embed_dim)]
        )

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        i, j = tf.range(n_dest)[:, None], tf.range(n_src)
        mask = tf.reshape(tf.cast(i >= j, dtype), [1, n_dest, n_src])
        return tf.tile(mask, [batch_size, 1, 1])

    def call(self, enc_out, target, training=False):
        causal_mask = self.causal_attention_mask(
            tf.shape(target)[0], tf.shape(target)[1], tf.shape(target)[1], tf.bool
        )
        target_att = self.self_att(
            target, target, attention_mask=causal_mask, training=training
        )
        target_norm = self.layernorm1(
            target + self.self_dropout(target_att, training=training)
        )
        enc_out_att = self.enc_att(target_norm, enc_out, training=training)
        enc_out_norm = self.layernorm2(
            target_norm + self.enc_dropout(enc_out_att, training=training)
        )
        ffn_out = self.ffn(enc_out_norm)
        return self.layernorm3(
            enc_out_norm + self.ffn_dropout(ffn_out, training=training)
        )


class Transformer(keras.Model):
    def __init__(
        self,
        num_hid,
        num_head,
        ff_dim,
        target_maxlen,
        num_layers_enc,
        num_layers_dec,
        num_classes,
        dropout_rate,
    ):
        super().__init__()
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.num_layers_enc, self.num_layers_dec = num_layers_enc, num_layers_dec
        self.target_maxlen, self.num_classes = target_maxlen, num_classes
        self.num_hid = num_hid
        self.enc_input = SpeechFeatureEmbedding(num_hid)
        self.dropout_enc_input = layers.Dropout(dropout_rate)
        self.dec_input = TokenEmbedding(num_classes, target_maxlen, num_hid)
        self.dropout_dec_input = layers.Dropout(dropout_rate)
        self.encoder = keras.Sequential(
            [self.enc_input, self.dropout_enc_input]
            + [
                TransformerEncoder(num_hid, num_head, ff_dim, dropout_rate)
                for _ in range(num_layers_enc)
            ]
        )
        for i in range(num_layers_dec):
            setattr(
                self,
                f"dec_layer_{i}",
                TransformerDecoder(num_hid, num_head, ff_dim, dropout_rate),
            )
        self.classifier = layers.Dense(num_classes)

    def decode(self, enc_out, target, training=False):
        y = self.dec_input(target)
        y = self.dropout_dec_input(y, training=training)
        for i in range(self.num_layers_dec):
            y = getattr(self, f"dec_layer_{i}")(enc_out, y, training=training)
        return y

    def call(self, inputs, training=False):
        source = self.encoder(inputs[0], training=training)
        target = self.decode(source, inputs[1], training=training)
        return self.classifier(target)

    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, batch):
        with tf.GradientTape() as tape:
            preds = self([batch["source"], batch["target"][:, :-1]], training=True)
            mask = tf.math.logical_not(tf.math.equal(batch["target"][:, 1:], 0))
            loss = self.compute_loss(
                y=batch["target"][:, 1:], y_pred=preds, sample_weight=mask
            )
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables)
        )
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def test_step(self, batch):
        preds = self([batch["source"], batch["target"][:, :-1]], training=False)
        mask = tf.math.logical_not(tf.math.equal(batch["target"][:, 1:], 0))
        loss = self.compute_loss(
            y=batch["target"][:, 1:], y_pred=preds, sample_weight=mask
        )
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def beam_search_decode(
        self, source, start_token_idx, end_token_idx, beam_width=3, max_len=None
    ):
        if max_len is None:
            max_len = self.target_maxlen - 1
        batch_size = tf.shape(source)[0]
        enc_output = self.encoder(source, training=False)
        beams = tf.ones((batch_size, 1), dtype=tf.int32) * start_token_idx
        scores = tf.zeros((batch_size, 1), dtype=tf.float32)
        for _ in range(max_len):
            num_beams = tf.shape(beams)[1]
            tiled_enc_output = tf.tile(
                tf.expand_dims(enc_output, 1), [1, num_beams, 1, 1]
            )
            tiled_enc_output = tf.reshape(
                tiled_enc_output, [batch_size * num_beams, -1, self.num_hid]
            )
            current_beams = tf.reshape(beams, [batch_size * num_beams, -1])
            preds = self.decode(tiled_enc_output, current_beams, training=False)
            logits = self.classifier(preds[:, -1, :])
            log_probs = tf.nn.log_softmax(logits)
            log_probs = tf.reshape(log_probs, [batch_size, num_beams, -1])
            total_scores = scores[:, :, None] + log_probs
            total_scores_flat = tf.reshape(total_scores, [batch_size, -1])
            top_scores, top_indices = tf.math.top_k(total_scores_flat, k=beam_width)
            beam_indices = top_indices // self.num_classes
            token_indices = top_indices % self.num_classes
            new_beams = tf.gather(beams, beam_indices, batch_dims=1)
            new_beams = tf.concat([new_beams, tf.expand_dims(token_indices, 2)], axis=2)
            beams = new_beams
            scores = top_scores
            if tf.reduce_any(tf.reduce_all(beams == end_token_idx, axis=2)):
                break
        return beams[:, 0, :]


class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, init_lr, warmup_lr, final_lr, warmup_epochs, decay_epochs, steps_per_epoch
    ):
        super().__init__()
        self.init_lr, self.warmup_lr, self.final_lr = [
            tf.cast(x, tf.float32) for x in [init_lr, warmup_lr, final_lr]
        ]
        self.warmup_epochs, self.decay_epochs, self.steps_per_epoch = [
            tf.cast(x, tf.float32)
            for x in [warmup_epochs, decay_epochs, steps_per_epoch]
        ]

    def __call__(self, step):
        epoch = tf.cast(step, tf.float32) / self.steps_per_epoch

        def warmup_fn():
            return self.init_lr + (epoch / self.warmup_epochs) * (
                self.warmup_lr - self.init_lr
            )

        def decay_fn():
            p = (epoch - self.warmup_epochs) / self.decay_epochs
            return tf.maximum(
                self.final_lr, self.warmup_lr - p * (self.warmup_lr - self.final_lr)
            )

        return tf.cond(epoch < self.warmup_epochs, warmup_fn, decay_fn)

    def get_config(self):
        return {
            k: float(v) for k, v in self.__dict__.items() if isinstance(v, tf.Tensor)
        }


# --- PART 3: THE BACKGROUND WORKER FOR TRAINING ---
class TrainingWorker(QObject):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.signals = WorkerSignals()
        self._is_running = True
        self.tensorboard_process = None

    def stop(self):
        self.signals.message.emit(
            "--- Stop signal received. Finishing current batch... ---"
        )
        self._is_running = False

    def run(self):
        try:
            self._setup_logging()
            logger = logging.getLogger(__name__)
            log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self._launch_tensorboard(log_dir)
            logger.info("Starting training process in background thread...")
            all_data, vectorizer = self._load_and_prepare_data()
            train_ds, val_ds = self._create_datasets(all_data, vectorizer)
            self._build_and_train_model(train_ds, val_ds, vectorizer, log_dir)
        except Exception as e:
            self.signals.error.emit((type(e), e, traceback.format_exc()))
        finally:
            self._shutdown_tensorboard()
            self.signals.finished.emit()

    def _launch_tensorboard(self, log_dir):
        logger = logging.getLogger(__name__)
        log_parent_dir = Path(log_dir).parent.as_posix()
        command = [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            log_parent_dir,
            "--port=0",
        ]
        logger.info(f"Launching TensorBoard with command: {' '.join(command)}")
        self.tensorboard_process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in iter(self.tensorboard_process.stdout.readline, ""):
            logger.info(f"[TensorBoard] {line.strip()}")
            if "http://localhost:" in line:
                match = re.search(r"(http://localhost:\d+)", line)
                if match:
                    url = match.group(1)
                    logger.info(f"TensorBoard running at: {url}")
                    self.signals.tensorboard_path.emit(url)
                    return
        logger.warning("Could not determine TensorBoard URL.")

    def _shutdown_tensorboard(self):
        if self.tensorboard_process:
            logger = logging.getLogger(__name__)
            logger.info("Shutting down TensorBoard process...")
            self.tensorboard_process.terminate()
            self.tensorboard_process.wait(timeout=5)
            logger.info("TensorBoard process terminated.")

    def _setup_logging(self):
        logger = logging.getLogger(__name__)
        if logger.hasHandlers():
            logger.handlers.clear()
        signal_handler = SignalLogHandler(self.signals)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        signal_handler.setFormatter(formatter)
        logger.addHandler(signal_handler)
        log_file_path = (
            Path(
                QStandardPaths.writableLocation(
                    QStandardPaths.StandardLocation.AppDataLocation
                )
            )
            / "ASRTrainer"
            / "logs"
        )
        log_file_path.mkdir(parents=True, exist_ok=True)
        log_file = (
            log_file_path / f"session_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

    def _load_and_prepare_data(self):
        logger = logging.getLogger(__name__)
        logger.info(f"Loading data from: {self.hparams.WAV_DIR}")
        all_data, unmatched = get_data(
            Path(self.hparams.WAV_DIR),
            Path(self.hparams.TRANSCRIPT_FILE),
            self.hparams.MAX_TARGET_LEN,
        )
        if not all_data:
            raise FileNotFoundError(
                f"No audio files were matched. Check transcript format/paths. Unmatched examples: {unmatched[:5]}"
            )
        logger.info(
            f"Found {len(all_data)} matched samples. {len(unmatched)} files were unmatched."
        )
        vectorizer = VectorizeChar(self.hparams.MAX_TARGET_LEN)
        logger.info(f"Vocabulary size: {len(vectorizer.get_vocabulary())}")
        return all_data, vectorizer

    def _create_datasets(self, all_data, vectorizer):
        logger = logging.getLogger(__name__)
        split_idx = int(len(all_data) * self.hparams.VALIDATION_SPLIT)
        train_data, val_data = all_data[:split_idx], all_data[split_idx:]
        logger.info(
            f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}"
        )
        train_ds = create_tf_dataset(
            train_data, vectorizer, self.hparams, self.hparams.BATCH_SIZE
        )
        val_ds = create_tf_dataset(
            val_data, vectorizer, self.hparams, self.hparams.BATCH_SIZE
        )
        return train_ds, val_ds

    def _build_and_train_model(self, train_ds, val_ds, vectorizer, log_dir):
        logger = logging.getLogger(__name__)
        display_batch = next(iter(val_ds))
        checkpoint_path = Path(log_dir) / "checkpoints" / "best_model.weights.h5"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks = [
            GUIDisplayOutputs(
                display_batch, vectorizer, self.signals, self.hparams.BEAM_WIDTH
            ),
            EarlyStopping(
                monitor="val_loss", patience=10, verbose=1, restore_best_weights=True
            ),
            StopTrainingCallback(self),
            keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
            ProgressCallback(self.signals),
            ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=True,
                monitor="val_loss",
                mode="min",
                save_best_only=True,
            ),
        ]
        model = Transformer(
            num_hid=self.hparams.NUM_HID,
            num_head=self.hparams.NUM_HEAD,
            ff_dim=self.hparams.FF_DIM,
            target_maxlen=self.hparams.MAX_TARGET_LEN,
            num_layers_enc=self.hparams.NUM_LAYERS_ENC,
            num_layers_dec=self.hparams.NUM_LAYERS_DEC,
            num_classes=len(vectorizer.get_vocabulary()),
            dropout_rate=self.hparams.DROPOUT_RATE,
        )
        lr_schedule = CustomSchedule(
            self.hparams.INIT_LR,
            self.hparams.WARMUP_LR,
            self.hparams.FINAL_LR,
            self.hparams.WARMUP_EPOCHS,
            self.hparams.DECAY_EPOCHS,
            len(train_ds),
        )
        model.compile(
            optimizer=keras.optimizers.Adam(lr_schedule),
            loss=keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction="none"
            ),
        )
        logger.info("Model compiled successfully. Starting training...")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.hparams.EPOCHS,
            callbacks=callbacks,
            verbose=0,
        )
        save_path = getattr(self.hparams, "MODEL_SAVE_PATH", None)
        if self._is_running and save_path:
            logger.info(f"Training complete. Saving best model to: {save_path}")
            try:
                model.save(save_path)
                logger.info("Model saved successfully.")
            except Exception as e:
                logger.error(f"Failed to save model. Error: {e}")
        elif not self._is_running:
            logger.info("Training was stopped by user. Model was not saved.")
        else:
            logger.info(
                "Training finished, but no save path was provided. Model was not saved."
            )


# --- PART 4: MAIN PYQT6 GUI APPLICATION ---
class ASRTrainingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASR Model Training Dashboard")
        self.params_widgets = {}
        self.thread, self.worker = None, None
        self.load_settings()
        self._setup_ui()
        self.statusBar().showMessage(
            "Ready. Please select data paths and start training."
        )

    def _setup_ui(self):
        self._create_menu_bar()
        self.setStatusBar(QStatusBar(self))
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.splitter, 1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.splitter.addWidget(left_panel)
        paths_groupbox = QGroupBox("Data Paths")
        self._create_path_widgets(paths_groupbox)
        left_layout.addWidget(paths_groupbox)
        params_groupbox = QGroupBox("Hyperparameters")
        self._create_param_widgets(params_groupbox)
        left_layout.addWidget(params_groupbox)
        left_layout.addStretch()

        self.right_tabs = QTabWidget()
        self.splitter.addWidget(self.right_tabs)
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self.output_text = QTextEdit()
        self.output_text.setHtml(
            "<h3>Welcome!</h3><p>Please select your data paths and configure hyperparameters on the left, then click 'Start Training'.</p>"
        )
        self.output_text.setReadOnly(True)
        log_layout.addWidget(self.output_text)
        self.right_tabs.addTab(log_tab, "Live Logs")
        tensorboard_tab = QWidget()
        tensorboard_layout = QVBoxLayout(tensorboard_tab)
        self.tensorboard_view = QWebEngineView()
        self.tensorboard_view.setHtml(
            "<h1>TensorBoard</h1><p>Start a training run to activate this view.</p>"
        )
        tensorboard_layout.addWidget(self.tensorboard_view)
        self.right_tabs.addTab(tensorboard_tab, "TensorBoard")
        inference_tab = QWidget()
        self._create_inference_tab_widgets(inference_tab)
        self.right_tabs.addTab(inference_tab, "Inference")

        bottom_controls_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setToolTip(
            "Shows the progress of the current training epoch."
        )
        self.progress_bar.hide()
        bottom_controls_layout.addWidget(self.progress_bar)
        bottom_controls_layout.addStretch()
        self.control_button = QPushButton("Start Training")
        self.control_button.clicked.connect(self.toggle_training)
        self.control_button.setToolTip("Start or Stop the training process.")
        bottom_controls_layout.addWidget(
            self.control_button, alignment=Qt.AlignmentFlag.AlignRight
        )
        main_layout.addLayout(bottom_controls_layout)

        settings = QSettings("MyASRApp", "TrainingDashboard")
        splitter_state = settings.value("splitterSizes")
        if splitter_state:
            self.splitter.restoreState(splitter_state)
        else:
            self.splitter.setSizes([400, 800])
        self.wav_dir_edit.setText(settings.value("wav_dir", ""))
        self.transcript_file_edit.setText(settings.value("transcript_file", ""))
        self.model_save_path_edit.setText(settings.value("model_save_path", ""))
        theme = settings.value("theme", "charcoal")
        if theme == "android":
            self.apply_android_theme()
        elif theme == "arctic":
            self.apply_arctic_theme()
        elif theme == "solarized":
            self.apply_solarized_theme()
        elif theme == "nord":
            self.apply_nord_theme()
        else:
            self.apply_charcoal_theme()

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        save_config_action = QAction("Save Configuration...", self)
        save_config_action.triggered.connect(self.save_configuration)
        file_menu.addAction(save_config_action)
        load_config_action = QAction("Load Configuration...", self)
        load_config_action.triggered.connect(self.load_configuration)
        file_menu.addAction(load_config_action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        view_menu = menu_bar.addMenu("View")
        theme_submenu = view_menu.addMenu("Themes")
        charcoal_action = QAction("Charcoal Black", self)
        charcoal_action.triggered.connect(self.apply_charcoal_theme)
        theme_submenu.addAction(charcoal_action)
        android_action = QAction("Classic Android", self)
        android_action.triggered.connect(self.apply_android_theme)
        theme_submenu.addAction(android_action)
        theme_submenu.addSeparator()
        arctic_action = QAction("Arctic Light", self)
        arctic_action.triggered.connect(self.apply_arctic_theme)
        theme_submenu.addAction(arctic_action)
        solarized_action = QAction("Solarized Dark", self)
        solarized_action.triggered.connect(self.apply_solarized_theme)
        theme_submenu.addAction(solarized_action)
        nord_action = QAction("Nord", self)
        nord_action.triggered.connect(self.apply_nord_theme)
        theme_submenu.addAction(nord_action)
        training_menu = menu_bar.addMenu("Training")
        self.start_stop_action = QAction("Start Training", self)
        self.start_stop_action.triggered.connect(self.toggle_training)
        training_menu.addAction(self.start_stop_action)
        help_menu = menu_bar.addMenu("Help")
        links_menu = help_menu.addMenu("Online Resources")
        github_action = QAction("GitHub Repository", self)
        github_action.triggered.connect(
            lambda: self.open_link("https://github.com/YOUR_USERNAME/YOUR_REPO")
        )
        links_menu.addAction(github_action)
        issue_action = QAction("Report an Issue", self)
        issue_action.triggered.connect(
            lambda: self.open_link("https://github.com/YOUR_USERNAME/YOUR_REPO/issues")
        )
        links_menu.addAction(issue_action)
        help_menu.addSeparator()
        view_logs_action = QAction("Open Logs Folder", self)
        view_logs_action.triggered.connect(self.open_logs_folder)
        help_menu.addAction(view_logs_action)
        help_menu.addSeparator()
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def _create_path_widgets(self, parent):
        layout = QGridLayout(parent)
        self.wav_dir_edit = QLineEdit()
        self.wav_dir_edit.setPlaceholderText("Select your 8kHz WAV folder...")
        self.wav_dir_edit.setToolTip(
            "Path to the folder containing your 8kHz .wav audio files."
        )
        self.transcript_file_edit = QLineEdit()
        self.transcript_file_edit.setPlaceholderText(
            "Select your transcript.txt file..."
        )
        self.transcript_file_edit.setToolTip(
            "Path to the transcript.txt file that maps filenames to text."
        )
        wav_browse_btn = QPushButton("Browse...")
        wav_browse_btn.clicked.connect(self.select_wav_directory)
        transcript_browse_btn = QPushButton("Browse...")
        transcript_browse_btn.clicked.connect(self.select_transcript_file)
        layout.addWidget(QLabel("WAV Directory:"), 0, 0)
        layout.addWidget(self.wav_dir_edit, 0, 1)
        layout.addWidget(wav_browse_btn, 0, 2)
        layout.addWidget(QLabel("Transcript File:"), 1, 0)
        layout.addWidget(self.transcript_file_edit, 1, 1)
        layout.addWidget(transcript_browse_btn, 1, 2)
        self.model_save_path_edit = QLineEdit()
        self.model_save_path_edit.setPlaceholderText(
            "(Optional) Select location to save model..."
        )
        self.model_save_path_edit.setToolTip(
            "Select where to save the final trained model.\nSaved in .keras format."
        )
        save_path_browse_btn = QPushButton("Browse...")
        save_path_browse_btn.clicked.connect(self.select_model_save_path)
        layout.addWidget(QLabel("Model Save Path:"), 2, 0)
        layout.addWidget(self.model_save_path_edit, 2, 1)
        layout.addWidget(save_path_browse_btn, 2, 2)

    def _create_param_widgets(self, parent):
        layout = QGridLayout(parent)
        int_validator = QIntValidator(1, 9999)
        float_validator = QDoubleValidator(0.0, 1.0, 8)
        params_to_show = {
            "NUM_HID": HParams.NUM_HID,
            "NUM_HEAD": HParams.NUM_HEAD,
            "FF_DIM": HParams.FF_DIM,
            "NUM_LAYERS_ENC": HParams.NUM_LAYERS_ENC,
            "NUM_LAYERS_DEC": HParams.NUM_LAYERS_DEC,
            "DROPOUT_RATE": HParams.DROPOUT_RATE,
            "BATCH_SIZE": HParams.BATCH_SIZE,
            "EPOCHS": HParams.EPOCHS,
            "INIT_LR": f"{HParams.INIT_LR:.0e}",
            "WARMUP_LR": f"{HParams.WARMUP_LR:.0e}",
        }
        tooltips = {
            "NUM_HID": "The main hidden dimension for embeddings and transformers.",
            "NUM_HEAD": "Number of attention heads in Multi-Head Attention.",
            "FF_DIM": "Inner dimension of the feed-forward network in transformers.",
            "NUM_LAYERS_ENC": "Number of layers in the Transformer encoder.",
            "NUM_LAYERS_DEC": "Number of layers in the Transformer decoder.",
            "DROPOUT_RATE": "Dropout rate for regularization (e.g., 0.2).",
            "BATCH_SIZE": "Number of samples per training batch. Lower if you run out of memory.",
            "EPOCHS": "Maximum number of epochs to train for. Early stopping will likely stop it sooner.",
            "INIT_LR": "Initial learning rate for the warmup phase.",
            "WARMUP_LR": "Peak learning rate after the warmup phase.",
        }
        for row, (name, value) in enumerate(params_to_show.items()):
            label = QLabel(f"{name}:")
            label.setToolTip(tooltips.get(name, ""))
            layout.addWidget(label, row, 0)
            entry = QLineEdit(str(value))
            entry.setToolTip(tooltips.get(name, ""))
            if name in [
                "NUM_HID",
                "NUM_HEAD",
                "FF_DIM",
                "NUM_LAYERS_ENC",
                "NUM_LAYERS_DEC",
                "BATCH_SIZE",
                "EPOCHS",
            ]:
                entry.setValidator(int_validator)
            elif name in ["DROPOUT_RATE"]:
                entry.setValidator(float_validator)
            layout.addWidget(entry, row, 1)
            self.params_widgets[name] = entry

    def _create_inference_tab_widgets(self, parent_tab):
        layout = QVBoxLayout(parent_tab)
        model_group = QGroupBox("1. Load Trained Model")
        model_layout = QHBoxLayout()
        self.inference_model_path_edit = QLineEdit()
        self.inference_model_path_edit.setPlaceholderText(
            "Select a saved .keras model folder..."
        )
        self.inference_model_path_edit.setReadOnly(True)
        load_model_btn = QPushButton("Load Model")
        load_model_btn.clicked.connect(self.load_inference_model)
        model_layout.addWidget(self.inference_model_path_edit)
        model_layout.addWidget(load_model_btn)
        model_group.setLayout(model_layout)
        audio_group = QGroupBox("2. Select Audio File for Transcription")
        audio_layout = QHBoxLayout()
        self.inference_audio_path_edit = QLineEdit()
        self.inference_audio_path_edit.setPlaceholderText(
            "Select a .wav file to transcribe..."
        )
        self.inference_audio_path_edit.setReadOnly(True)
        select_audio_btn = QPushButton("Select Audio")
        select_audio_btn.clicked.connect(self.select_inference_audio)
        audio_layout.addWidget(self.inference_audio_path_edit)
        audio_layout.addWidget(select_audio_btn)
        audio_group.setLayout(audio_layout)
        self.transcribe_button = QPushButton("Transcribe")
        self.transcribe_button.setToolTip(
            "Run the loaded model on the selected audio file."
        )
        self.transcribe_button.clicked.connect(self.run_transcription)
        self.transcribe_button.setEnabled(False)
        result_group = QGroupBox("3. Transcription Result")
        result_layout = QVBoxLayout()
        self.inference_result_text = QTextEdit()
        self.inference_result_text.setReadOnly(True)
        self.inference_result_text.setMinimumHeight(100)
        result_layout.addWidget(self.inference_result_text)
        result_group.setLayout(result_layout)
        layout.addWidget(model_group)
        layout.addWidget(audio_group)
        layout.addWidget(self.transcribe_button)
        layout.addWidget(result_group)
        layout.addStretch()
        self.inference_model = None
        self.inference_hparams = HParams()

    def load_inference_model(self):
        model_path = QFileDialog.getExistingDirectory(
            self, "Select Saved Keras Model Folder"
        )
        if not model_path:
            return
        try:
            self.statusBar().showMessage("Loading model...")
            self.inference_model = keras.models.load_model(model_path)
            self.inference_model_path_edit.setText(model_path)
            self.statusBar().showMessage(
                f"Successfully loaded model from {Path(model_path).name}", 5000
            )
            self.log_to_gui(f"INFO: Inference model loaded from {model_path}")
            self._check_transcribe_button_state()
        except Exception as e:
            QMessageBox.critical(
                self, "Error Loading Model", f"Could not load the model.\n\nError: {e}"
            )
            self.inference_model = None
            self._check_transcribe_button_state()

    def select_inference_audio(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "WAV Files (*.wav)"
        )
        if filename:
            self.inference_audio_path_edit.setText(filename)
            self._check_transcribe_button_state()

    def _check_transcribe_button_state(self):
        self.transcribe_button.setEnabled(
            self.inference_model is not None
            and bool(self.inference_audio_path_edit.text())
        )

    def run_transcription(self):
        self.statusBar().showMessage("Transcribing audio...")
        self.transcribe_button.setEnabled(False)
        QApplication.processEvents()
        try:
            audio_path = self.inference_audio_path_edit.text()
            audio_tensor = path_to_audio(audio_path, self.inference_hparams)
            audio_tensor = tf.expand_dims(audio_tensor, axis=0)
            temp_vectorizer = VectorizeChar(self.inference_hparams.MAX_TARGET_LEN)
            start_idx = temp_vectorizer.char_to_idx["<"]
            end_idx = temp_vectorizer.char_to_idx[">"]
            output_array = self.inference_model.beam_search_decode(
                audio_tensor, start_idx, end_idx
            )
            prediction = temp_vectorizer.decode(output_array.numpy()[0])
            self.inference_result_text.setText(prediction)
            self.statusBar().showMessage("Transcription complete.", 5000)
        except Exception as e:
            self.inference_result_text.setText(f"ERROR:\n{e}")
            self.statusBar().showMessage(
                "An error occurred during transcription.", 5000
            )
        finally:
            self.transcribe_button.setEnabled(True)

    def select_wav_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select WAV File Directory")
        if not directory:
            return
        self.wav_dir_edit.setText(directory)
        potential_transcript_path = Path(directory).parent / "transcript.txt"
        if potential_transcript_path.exists():
            self.transcript_file_edit.setText(str(potential_transcript_path))
            self.statusBar().showMessage("Auto-detected transcript.txt", 3000)

    def select_transcript_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Transcript File", "", "Text Files (*.txt);;All Files (*)"
        )
        if filename:
            self.transcript_file_edit.setText(filename)

    def select_model_save_path(self):
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Select Model Save Location",
            "",
            "Keras Model (*.keras);;HDF5 Model (*.h5)",
        )
        if filename:
            self.model_save_path_edit.setText(filename)

    def save_configuration(self):
        config_path, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "", "JSON Files (*.json)"
        )
        if not config_path:
            return
        config_data = {
            name: widget.text() for name, widget in self.params_widgets.items()
        }
        config_data["WAV_DIR"] = self.wav_dir_edit.text()
        config_data["TRANSCRIPT_FILE"] = self.transcript_file_edit.text()
        config_data["MODEL_SAVE_PATH"] = self.model_save_path_edit.text()
        try:
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=4)
            self.statusBar().showMessage(
                f"Configuration saved to {Path(config_path).name}", 5000
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to save configuration file.\n\nError: {e}"
            )

    def load_configuration(self):
        config_path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", "JSON Files (*.json)"
        )
        if not config_path:
            return
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            for name, value in config_data.items():
                if name in self.params_widgets:
                    self.params_widgets[name].setText(value)
            self.wav_dir_edit.setText(config_data.get("WAV_DIR", ""))
            self.transcript_file_edit.setText(config_data.get("TRANSCRIPT_FILE", ""))
            self.model_save_path_edit.setText(config_data.get("MODEL_SAVE_PATH", ""))
            self.statusBar().showMessage(
                f"Configuration loaded from {Path(config_path).name}", 5000
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load or parse configuration file.\n\nError: {e}",
            )

    def open_logs_folder(self):
        log_path = Path("logs")
        if not log_path.exists():
            QMessageBox.information(
                self,
                "No Logs Found",
                "The 'logs' directory has not been created yet. Please run a training session first.",
            )
            return
        try:
            if sys.platform == "win32":
                os.startfile(log_path.resolve())
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(log_path.resolve())])
            else:
                subprocess.Popen(["xdg-open", str(log_path.resolve())])
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Could not open the logs folder.\n\nError: {e}"
            )

    def open_link(self, url_string):
        QDesktopServices.openUrl(QUrl(url_string))

    def show_about_dialog(self):
        QMessageBox.about(
            self,
            "About ASR Training Dashboard",
            """
        <h2>ASR Model Training Dashboard v2.0</h2>
        <p>This application allows you to train a Transformer-based Automatic Speech Recognition (ASR) model, test it, and monitor it with TensorBoard, all in one place.</p>
        
        <h3>How to Use:</h3>
        <ol>
            <li><b>Load Data:</b> Use the 'Browse...' buttons in the 'Data Paths' section to select your audio folder and transcript file.</li>
            <li><b>Configure Training:</b> Adjust hyperparameters or load a saved configuration from the File menu.</li>
            <li><b>Start Training:</b> Click the "Start Training" button.</li>
            <li><b>Monitor:</b> Watch the 'Live Logs' and 'TensorBoard' tabs for progress.</li>
            <li><b>Test Model:</b> After training, go to the 'Inference' tab to load your saved model and transcribe single audio files.</li>
        </ol>

        <p>Developed with PyQt6 and TensorFlow.</p>
        """,
        )

    def apply_charcoal_theme(self):
        self.setStyleSheet(self.get_charcoal_stylesheet())
        self.setProperty("theme", "charcoal")
        self.statusBar().showMessage("Theme changed to Charcoal Black.", 3000)

    def apply_android_theme(self):
        self.setStyleSheet(self.get_android_stylesheet())
        self.setProperty("theme", "android")
        self.statusBar().showMessage("Theme changed to Classic Android.", 3000)

    def apply_arctic_theme(self):
        self.setStyleSheet(self.get_arctic_stylesheet())
        self.setProperty("theme", "arctic")
        self.statusBar().showMessage("Theme changed to Arctic Light.", 3000)

    def apply_solarized_theme(self):
        self.setStyleSheet(self.get_solarized_stylesheet())
        self.setProperty("theme", "solarized")
        self.statusBar().showMessage("Theme changed to Solarized Dark.", 3000)

    def apply_nord_theme(self):
        self.setStyleSheet(self.get_nord_stylesheet())
        self.setProperty("theme", "nord")
        self.statusBar().showMessage("Theme changed to Nord.", 3000)

    def get_charcoal_stylesheet(self):
        return """
        QWidget {font-family: Lato, "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 14px; background-color: #212121; color: #DCDCDC;}
        QMainWindow, QSplitter::handle {background-color: #212121;}
        QGroupBox {border: 1px solid #424242; border-radius: 5px; margin-top: 1ex; font-weight: bold;}
        QGroupBox::title {subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px;}
        QTextEdit, QLineEdit {background-color: #313131; color: #DCDCDC; border: 1px solid #424242; border-radius: 4px; padding: 5px;}
        QPushButton {background-color: #007ACC; color: white; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold;}
        QPushButton:hover {background-color: #0099FF;}
        QPushButton:pressed {background-color: #005A9E;}
        QPushButton:disabled {background-color: #424242; color: #888888;}
        QProgressBar {border: 1px solid #424242; border-radius: 4px; text-align: center; color: #DCDCDC;}
        QProgressBar::chunk {background-color: #007ACC; border-radius: 4px;}
        QMenuBar {background-color: #313131; color: #DCDCDC;}
        QMenuBar::item:selected {background: #50555A;}
        QMenu {background-color: #313131; color: #DCDCDC; border: 1px solid #5A5A5A;}
        QMenu::item:selected {background-color: #007ACC; color: white;}
        QTabWidget::pane { border: 1px solid #424242; }
        QTabBar::tab { background: #313131; color: #DCDCDC; padding: 8px 20px; border: 1px solid #424242; border-bottom: none; }
        QTabBar::tab:selected { background: #50555A; }
        """

    def get_android_stylesheet(self):
        return """
        QWidget {font-family: Lato, "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 14px; background-color: #282C34; color: #DCDCDC;}
        QMainWindow, QSplitter::handle {background-color: #282C34;}
        QGroupBox {border: 1px solid #50555A; border-radius: 5px; margin-top: 1ex; font-weight: bold;}
        QGroupBox::title {subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px;}
        QTextEdit, QLineEdit {background-color: #383C44; color: #DCDCDC; border: 1px solid #50555A; border-radius: 4px; padding: 5px;}
        QPushButton {background-color: #33B5E5; color: #111111; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold;}
        QPushButton:hover {background-color: #00DDFF;}
        QPushButton:pressed {background-color: #0099CC;}
        QPushButton:disabled {background-color: #424242; color: #888888;}
        QProgressBar {border: 1px solid #50555A; border-radius: 4px; text-align: center; color: #DCDCDC;}
        QProgressBar::chunk {background-color: #33B5E5; border-radius: 4px;}
        QMenuBar {background-color: #383C44; color: #DCDCDC;}
        QMenuBar::item:selected {background: #50555A;}
        QMenu {background-color: #383C44; color: #DCDCDC; border: 1px solid #5A5A5A;}
        QMenu::item:selected {background-color: #33B5E5; color: #111111;}
        QTabWidget::pane { border: 1px solid #50555A; }
        QTabBar::tab { background: #383C44; color: #DCDCDC; padding: 8px 20px; border: 1px solid #50555A; border-bottom: none; }
        QTabBar::tab:selected { background: #50555A; }
        """

    def get_arctic_stylesheet(self):
        return """
        QWidget {font-family: Lato, "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 14px; background-color: #F5F5F5; color: #212121;}
        QMainWindow, QSplitter::handle {background-color: #F5F5F5;}
        QGroupBox {border: 1px solid #DCDCDC; border-radius: 5px; margin-top: 1ex; font-weight: bold;}
        QGroupBox::title {subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px;}
        QTextEdit, QLineEdit {background-color: #FFFFFF; color: #212121; border: 1px solid #DCDCDC; border-radius: 4px; padding: 5px;}
        QPushButton {background-color: #007ACC; color: white; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold;}
        QPushButton:hover {background-color: #0099FF;}
        QPushButton:pressed {background-color: #005A9E;}
        QPushButton:disabled {background-color: #E0E0E0; color: #AAAAAA;}
        QProgressBar {border: 1px solid #DCDCDC; border-radius: 4px; text-align: center; color: #212121;}
        QProgressBar::chunk {background-color: #007ACC; border-radius: 4px;}
        QMenuBar {background-color: #FFFFFF; color: #212121; border-bottom: 1px solid #DCDCDC;}
        QMenuBar::item:selected {background: #E0E0E0;}
        QMenu {background-color: #FFFFFF; color: #212121; border: 1px solid #DCDCDC;}
        QMenu::item:selected {background-color: #007ACC; color: white;}
        QTabWidget::pane { border-top: 1px solid #DCDCDC; }
        QTabBar::tab { background: #E0E0E0; color: #212121; padding: 8px 20px; border: 1px solid #DCDCDC; border-bottom: none; }
        QTabBar::tab:selected { background: #F5F5F5; }
        """

    def get_solarized_stylesheet(self):
        return """
        QWidget {font-family: "Menlo", "Consolas", monospace; font-size: 14px; background-color: #002b36; color: #839496;}
        QMainWindow, QSplitter::handle {background-color: #002b36;}
        QGroupBox {border: 1px solid #586e75; border-radius: 5px; margin-top: 1ex; font-weight: bold;}
        QGroupBox::title {subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px;}
        QTextEdit, QLineEdit {background-color: #073642; color: #839496; border: 1px solid #586e75; border-radius: 4px; padding: 5px;}
        QPushButton {background-color: #268bd2; color: #fdf6e3; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold;}
        QPushButton:hover {background-color: #1e72b3;}
        QPushButton:pressed {background-color: #1a629b;}
        QPushButton:disabled {background-color: #073642; color: #586e75;}
        QProgressBar {border: 1px solid #586e75; border-radius: 4px; text-align: center; color: #fdf6e3;}
        QProgressBar::chunk {background-color: #268bd2; border-radius: 4px;}
        QMenuBar {background-color: #073642; color: #839496;}
        QMenuBar::item:selected {background: #586e75;}
        QMenu {background-color: #073642; color: #839496; border: 1px solid #586e75;}
        QMenu::item:selected {background-color: #268bd2; color: #fdf6e3;}
        QTabWidget::pane { border: 1px solid #586e75; }
        QTabBar::tab { background: #073642; color: #839496; padding: 8px 20px; border: 1px solid #586e75; border-bottom: none; }
        QTabBar::tab:selected { background: #002b36; }
        """

    def get_nord_stylesheet(self):
        return """
        QWidget {font-family: Lato, "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 14px; background-color: #2E3440; color: #D8DEE9;}
        QMainWindow, QSplitter::handle {background-color: #2E3440;}
        QGroupBox {border: 1px solid #4C566A; border-radius: 5px; margin-top: 1ex; font-weight: bold;}
        QGroupBox::title {subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px;}
        QTextEdit, QLineEdit {background-color: #3B4252; color: #D8DEE9; border: 1px solid #4C566A; border-radius: 4px; padding: 5px;}
        QPushButton {background-color: #88C0D0; color: #2E3440; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold;}
        QPushButton:hover {background-color: #96d0e1;}
        QPushButton:pressed {background-color: #7aaebb;}
        QPushButton:disabled {background-color: #4C566A; color: #69707e;}
        QProgressBar {border: 1px solid #4C566A; border-radius: 4px; text-align: center; color: #2E3440;}
        QProgressBar::chunk {background-color: #88C0D0; border-radius: 4px;}
        QMenuBar {background-color: #3B4252; color: #D8DEE9;}
        QMenuBar::item:selected {background: #4C566A;}
        QMenu {background-color: #3B4252; color: #D8DEE9; border: 1px solid #4C566A;}
        QMenu::item:selected {background-color: #88C0D0; color: #2E3440;}
        QTabWidget::pane { border: 1px solid #4C566A; }
        QTabBar::tab { background: #3B4252; color: #D8DEE9; padding: 8px 20px; border: 1px solid #4C566A; border-bottom: none; }
        QTabBar::tab:selected { background: #434C5E; }
        """

    def toggle_training(self):
        if self.thread and self.thread.isRunning():
            self.stop_training()
        else:
            self.start_training()

    def start_training(self):
        wav_dir = self.wav_dir_edit.text()
        transcript_file = self.transcript_file_edit.text()
        if not wav_dir or not transcript_file:
            QMessageBox.warning(
                self,
                "Missing Paths",
                "Please select both a WAV directory and a transcript file before starting.",
            )
            return
        self.control_button.setText("Stop Training")
        self.start_stop_action.setText("Stop Training")
        self.control_button.setEnabled(True)
        self.output_text.clear()
        self.tensorboard_view.setHtml(
            "<h1>TensorBoard</h1><p>Starting training and launching TensorBoard...</p>"
        )
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.statusBar().showMessage("Starting training process...")
        try:
            gui_hparams = HParams()
            gui_hparams.WAV_DIR = wav_dir
            gui_hparams.TRANSCRIPT_FILE = transcript_file
            gui_hparams.MODEL_SAVE_PATH = self.model_save_path_edit.text()
            for name, widget in self.params_widgets.items():
                setattr(
                    gui_hparams,
                    name,
                    float(widget.text())
                    if "." in widget.text() or "e" in widget.text()
                    else int(widget.text()),
                )
        except ValueError as e:
            self.log_to_gui(f"ERROR: Invalid parameter value. Details: {e}")
            self.on_training_finished()
            return
        self.thread = QThread()
        self.worker = TrainingWorker(gui_hparams)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.signals.finished.connect(self.on_training_finished)
        self.worker.signals.message.connect(self.log_to_gui)
        self.worker.signals.error.connect(self.on_training_error)
        self.worker.signals.progress.connect(self.progress_bar.setValue)
        self.worker.signals.tensorboard_path.connect(self.load_tensorboard_url)
        self.thread.start()

    def stop_training(self):
        if self.worker:
            self.worker.stop()
            self.control_button.setText("Stopping...")
            self.start_stop_action.setText("Stopping...")
            self.control_button.setEnabled(False)
            self.start_stop_action.setEnabled(False)

    def on_training_finished(self):
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        self.control_button.setText("Start Training")
        self.start_stop_action.setText("Start Training")
        self.control_button.setEnabled(True)
        self.start_stop_action.setEnabled(True)
        self.progress_bar.hide()
        self.tensorboard_view.setHtml(
            "<h1>TensorBoard</h1><p>Training finished. Start a new run to activate this view.</p>"
        )
        self.statusBar().showMessage("Training finished. Ready for next run.")
        self.log_to_gui("\n--- Training session finished. ---")

    def load_tensorboard_url(self, url):
        self.tensorboard_view.setUrl(QUrl(url))
        self.statusBar().showMessage(
            "TensorBoard is running. See 'TensorBoard' tab.", 5000
        )

    def log_to_gui(self, message):
        self.output_text.append(message)

    def on_training_error(self, error_tuple):
        self.log_to_gui(f"\n--- FATAL ERROR in training thread ---\n{error_tuple[2]}")

    def save_settings(self):
        settings = QSettings("MyASRApp", "TrainingDashboard")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("splitterSizes", self.splitter.saveState())
        settings.setValue("wav_dir", self.wav_dir_edit.text())
        settings.setValue("transcript_file", self.transcript_file_edit.text())
        settings.setValue("model_save_path", self.model_save_path_edit.text())
        settings.setValue("theme", self.property("theme"))

    def load_settings(self):
        self.settings = QSettings("MyASRApp", "TrainingDashboard")

    def closeEvent(self, event):
        self.save_settings()
        if self.thread and self.thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "A training process is currently running. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.stop_training()
                self.thread.wait(5000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ASRTrainingGUI()
    window.show()
    sys.exit(app.exec())
