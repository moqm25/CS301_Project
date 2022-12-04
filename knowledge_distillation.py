import os
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import random
import distilled_model as kd_model
import group_10__semantic_segmentation_of_satellite_imagery as mdl
import simple_multi_unet_model as unet_model

class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)

            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            distillation_loss = (
                self.distillation_loss_fn(
                    tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                    tf.nn.softmax(student_predictions / self.temperature, axis=1),
                )
                * self.temperature**2
            )

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results



def main():
    teacher_model = tf.keras.models.load_model('models/final_model.hdf5',
                    custom_objects={'dice_loss_plus_1focal_loss': unet_model.get_total_loss(),
                                    'jacard_coef':unet_model.jacard_coef})
    student_model = kd_model.multi_unet_model(4)
    
    teacher_model.summary()
    student_model.summary()

    x_train, x_test, y_train, y_test = mdl.getData()

    prob_thresholds = np.linspace(0, 1, num=1000).tolist()
    metrics = ['accuracy', tf.keras.metrics.Precision(name="precision", thresholds=prob_thresholds), tf.keras.metrics.Recall(name="recall", thresholds=prob_thresholds)]


    distiller = Distiller(student=student_model, teacher=teacher_model)
    distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=metrics,
        student_loss_fn=unet_model.get_total_loss(),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=10,
    )

    # Distill teacher to student
    distiller.fit(x_train, y_train, epochs=3)

    # Evaluate student on test dataset
    distiller.evaluate(x_test, y_test)

    # Testing student model on the best model based on validation set
    # student_model.load_model_from_file(args.checkpoint_dir)
    # student_model.run_inference(dataset)
    # else:
    #     teacher_model = model.BigModel(args, "teacher")
    #     teacher_model.start_session()
    #     teacher_model.train(dataset)

        # Testing teacher model on the best model based on validation set


if __name__ == '__main__':
    main()
    # INVOCATION

    # Teacher
    # python main.py --model_type teacher --checkpoint_dir teachercpt --num_steps 50

    # Student
    # python main.py --model_type student --checkpoint_dir studentcpt --num_steps 50 --gpu 0

    # Student
    # python main.py --model_type student --checkpoint_dir studentcpt --load_teacher_from_checkpoint true --load_teacher_checkpoint_dir teachercpt --num_steps 50