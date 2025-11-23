"""Message queue integration for async processing."""

import logging
import json
from typing import Callable, Any, Optional
from datetime import datetime

try:
    import pika
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False

logger = logging.getLogger(__name__)


class MessageQueue:
    """
    RabbitMQ message queue for async ML inference.

    Enables decoupling of request handling and prediction processing.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5672,
        username: str = "guest",
        password: str = "guest",
        queue_name: str = "ml_predictions"
    ):
        """
        Initialize message queue.

        Args:
            host: RabbitMQ host
            port: RabbitMQ port
            username: RabbitMQ username
            password: RabbitMQ password
            queue_name: Queue name
        """
        if not RABBITMQ_AVAILABLE:
            raise ImportError(
                "Pika is not installed. "
                "Install it with: pip install pika"
            )

        self.host = host
        self.port = port
        self.queue_name = queue_name

        # Setup connection
        credentials = pika.PlainCredentials(username, password)
        parameters = pika.ConnectionParameters(
            host=host,
            port=port,
            credentials=credentials
        )

        try:
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()

            # Declare queue
            self.channel.queue_declare(queue=queue_name, durable=True)

            logger.info(f"Connected to RabbitMQ at {host}:{port}")

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            self.connection = None
            self.channel = None

    def publish(self, message: dict, routing_key: Optional[str] = None) -> bool:
        """
        Publish a message to the queue.

        Args:
            message: Message dictionary
            routing_key: Routing key (defaults to queue_name)

        Returns:
            True if successful
        """
        if self.channel is None:
            logger.error("Not connected to RabbitMQ")
            return False

        try:
            routing_key = routing_key or self.queue_name

            # Add timestamp
            message['timestamp'] = datetime.utcnow().isoformat()

            # Serialize message
            body = json.dumps(message)

            # Publish
            self.channel.basic_publish(
                exchange='',
                routing_key=routing_key,
                body=body,
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Make message persistent
                    content_type='application/json'
                )
            )

            logger.debug(f"Published message to {routing_key}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False

    def consume(
        self,
        callback: Callable[[dict], Any],
        auto_ack: bool = False
    ) -> None:
        """
        Consume messages from the queue.

        Args:
            callback: Function to process messages
            auto_ack: Whether to auto-acknowledge messages
        """
        if self.channel is None:
            logger.error("Not connected to RabbitMQ")
            return

        def on_message(ch, method, properties, body):
            """Message handler."""
            try:
                # Parse message
                message = json.loads(body)

                # Process message
                callback(message)

                # Acknowledge message
                if not auto_ack:
                    ch.basic_ack(delivery_tag=method.delivery_tag)

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                # Reject and requeue message
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

        # Set QoS to process one message at a time
        self.channel.basic_qos(prefetch_count=1)

        # Start consuming
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=on_message,
            auto_ack=auto_ack
        )

        logger.info(f"Starting to consume messages from {self.queue_name}")
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Stopped consuming messages")
            self.channel.stop_consuming()

    def get_queue_size(self) -> int:
        """
        Get number of messages in queue.

        Returns:
            Number of messages
        """
        if self.channel is None:
            return 0

        try:
            method_frame = self.channel.queue_declare(
                queue=self.queue_name,
                passive=True
            )
            return method_frame.method.message_count

        except Exception as e:
            logger.error(f"Failed to get queue size: {e}")
            return 0

    def purge_queue(self) -> bool:
        """
        Purge all messages from queue.

        Returns:
            True if successful
        """
        if self.channel is None:
            return False

        try:
            self.channel.queue_purge(queue=self.queue_name)
            logger.info(f"Purged queue {self.queue_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to purge queue: {e}")
            return False

    def close(self) -> None:
        """Close connection."""
        if self.connection:
            self.connection.close()
            logger.info("Closed RabbitMQ connection")


class AsyncPredictionProcessor:
    """
    Async prediction processor using message queues.

    Decouples API requests from model predictions.
    """

    def __init__(
        self,
        model: Any,
        queue: MessageQueue,
        result_callback: Optional[Callable] = None
    ):
        """
        Initialize async processor.

        Args:
            model: ML model instance
            queue: MessageQueue instance
            result_callback: Callback for results
        """
        self.model = model
        self.queue = queue
        self.result_callback = result_callback

        self.processed_count = 0
        self.error_count = 0

    def submit_prediction(self, features: Any, request_id: str) -> bool:
        """
        Submit prediction request to queue.

        Args:
            features: Input features
            request_id: Unique request ID

        Returns:
            True if submitted successfully
        """
        message = {
            'request_id': request_id,
            'features': features
        }

        return self.queue.publish(message)

    def process_predictions(self) -> None:
        """Start processing predictions from queue."""
        def callback(message: dict):
            """Process prediction message."""
            try:
                request_id = message['request_id']
                features = message['features']

                # Make prediction
                result = self.model.predict_with_preprocessing(features)

                # Add request ID
                result['request_id'] = request_id

                # Send result
                if self.result_callback:
                    self.result_callback(result)

                self.processed_count += 1

                logger.info(f"Processed prediction for request {request_id}")

            except Exception as e:
                self.error_count += 1
                logger.error(f"Prediction processing error: {e}")

        self.queue.consume(callback)

    def get_stats(self) -> dict:
        """
        Get processing statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'error_rate': (
                self.error_count / max(self.processed_count, 1)
            ),
            'queue_size': self.queue.get_queue_size()
        }
