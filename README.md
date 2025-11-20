pip install pyspark
customerID,gender,age,tenure,balance,products,hasCrCard,isActiveMember,estimatedSalary,Churn
15634602,Female,42,2,0.00,1,1,1,50000,No
15647311,Male,34,5,120000.00,2,0,1,60000,Yes
15619304,Female,30,3,50000.00,2,1,0,70000,No
...

from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.classification import LogisticRegressionWithLBFGS

# Spark session
spark = SparkSession.builder \
    .appName("Simple MLlib Churn") \
    .master("local[*]") \
    .getOrCreate()

sc = spark.sparkContext

# Load CSV
data = sc.textFile("customer_churn.csv")

# Convert CSV rows → LabeledPoint
parsed = data.filter(lambda x: not x.startswith("customerID")) \
    .map(lambda line: line.split(",")) \
    .map(lambda f: LabeledPoint(
        1.0 if f[9] == "Yes" else 0.0,   # Churn label
        Vectors.dense([
            float(f[2]),   # age
            float(f[3]),   # tenure
            float(f[4]),   # balance
            float(f[5]),   # products
            float(f[6]),   # hasCrCard
            float(f[7]),   # isActiveMember
            float(f[8])    # estimatedSalary
        ])
    ))

# Train-test split
train, test = parsed.randomSplit([0.8, 0.2])

# Train model
model = LogisticRegressionWithLBFGS.train(train)

# Evaluate accuracy
pred_and_label = test.map(lambda p: (model.predict(p.features), p.label))
accuracy = pred_and_label.filter(lambda x: x[0] == x[1]).count() / float(test.count())

print("Model Accuracy:", accuracy)

spark.stop()

python churn_mllib.py




// Abstract class with abstract and concrete methods
abstract class Shape {
  // Abstract method (must be implemented by subclasses)
  def area: Double

  // Concrete method (common implementation for all shapes)
  def display(): Unit = {
    println(s"Area of the shape: $area")
  }
}

// Trait for coloring shapes
trait Colorable {
  def color: String
  def showColor(): Unit = {
    println(s"The shape color is: $color")
  }
}

// Concrete class extending abstract class and mixing a trait
class Circle(val radius: Double, val color: String) extends Shape with Colorable {
  // Implement abstract method
  def area: Double = math.Pi * radius * radius
}

// Another concrete class
class Rectangle(val length: Double, val width: Double, val color: String) extends Shape with Colorable {
  def area: Double = length * width
}

// Main object to run the demo
object AbstractAndTraitDemo {
  def main(args: Array[String]): Unit = {
    val circle = new Circle(5, "Red")
    val rectangle = new Rectangle(4, 6, "Blue")

    println("=== Circle ===")
    circle.display()
    circle.showColor()

    println("\n=== Rectangle ===")
    rectangle.display()
    rectangle.showColor()
  }
}

apiVersion: apps/v1
kind: Deployment
metadata:
  name: exp-node-deploy
spec:
  replicas: 2
  selector:
    matchLabels:
      app: exp-node
  template:
    metadata:
      labels:
        app: exp-node
    spec:
      containers:
        - name: exp-node
          image: exp:1.0    # your Docker image
          ports:
            - containerPort: 3030
---
apiVersion: v1
kind: Service
metadata:
  name: exp-node-service
spec:
  selector:
    app: exp-node
  type: NodePort
  ports:
    - protocol: TCP
      port: 8080         # external port
      targetPort: 3030   # container port
