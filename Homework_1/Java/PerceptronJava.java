import java.util.ArrayList;
import java.util.Collections;
import java.util.concurrent.ThreadLocalRandom;


public class PerceptronJava
{

	public static class  Point
	{
		double yn;
		double x;
		double y;

		public Point(double xcoord, double ycoord, double yn)
		{
			this.x = xcoord;
			this.y = ycoord;
			this.yn = yn;
		}

		public Point(double xcoord, double ycoord)
		{
			this.x = xcoord;
			this.y = ycoord;
		}

	}

	public static class Line
	{
		double x1, x2, y1, y2, m, b;

		public Line(double xcoord1, double ycoord1, double xcoord2, double ycoord2)
		{
			this.x1 = xcoord1;
			this.y1 = ycoord1;
			this.x2 = xcoord2;
			this.y2 = ycoord2;

			this.m = (this.y2 - this.y1) / (this.x2 - this.x1);
			this.b = this.y1 - (this.m * this.x1);
		}
	}

	public static double randomDouble(double min, double max)
	{
		//double randomValue = min + (max - min) * rand.nextDouble();
		return (ThreadLocalRandom.current().nextDouble() * (max - min)) + min;
	}


	public static void RunSimulation(int trials, int numOfPoints)
	{
		long iterations = 0;
		double rateSum = 0.0;
		long count;
		long[] counts = new long[trials];
		double[] rates = new double[trials];

		for (int i = 0; i < trials; i++)
		{
			double[] weights = new double[]{0.0, 0.0, 0.0};
			count = 0;

			//Create line to initialize the point data
			double x1 = randomDouble(-1.0, 1.0);
			double x2 = randomDouble(-1.0, 1.0);
			double y1 = randomDouble(-1.0, 1.0);
			double y2 = randomDouble(-1.0, 1.0);


			Line line = new Line(x1, y1, x2, y2);

			ArrayList<Point> points = new ArrayList();

			//populate the array with points and their respective +/-
			for(int j = 0; j < numOfPoints; j++)
			{
				double newX, newY;
				newX = randomDouble(-1.0, 1.0);
				newY = randomDouble(-1.0, 1.0);

				Point p = new Point(newX, newY, 0.0);

				double lineY = (line.m * p.x) + line.b;

				if(p.y > lineY)
					p.yn = 1.0;
				else
					p.yn = -1.0;

				points.add(p);
			}

			//run PLA on points
			boolean mistake;
			do
			{
				mistake = false;
				for(int k = 0; k < numOfPoints; k++)
				{
					Point p_k = points.get(k);
					double actualY = p_k.yn;
					double dotProd = weights[0] + (weights[1] * p_k.x) + (weights[2] * p_k.y);
					double calcY;

					if (dotProd > 0)
						calcY = 1.0;
					else
						calcY = -1.0;

					if (actualY != calcY)
					{
						mistake = true;
						count ++;

						weights[0] +=  actualY;
						weights[1] +=  (actualY * p_k.x);
						weights[2] +=  (actualY * p_k.y);

						Collections.shuffle(points);

						break;
					}
				}
			} while(mistake);

			counts[i] = count;
			//Test Probability using the final trial weights, and last initialized Line object
			int errors = 0;
			for (int i1 = 0; i1 < 100; i1 ++)
			{
				double randX1 = randomDouble(-1.0, 1.0);
				double randX2 = randomDouble(-1.0, 1.0);

				double targetY = (line.m * randX1) + line.b;
				double randY;

				if(randX2 > targetY)
					randY = 1.0;
				else
					randY = -1.0;

				double dotProd = weights[0] + (weights[1] * randX1) + (weights[2] * randX2);
				double guessY;
				if (dotProd > 0)
					guessY = 1.0;
				else
					guessY = -1.0;

				if (randY != guessY)
					errors++;


			}
			double rate =  (double)errors / 100.0;
			rates[i] = rate;

		}


		for(int l = 0; l < trials; l++)
		{
			iterations += counts[l];
			rateSum += rates[l];
		}

		System.out.println("Average Iterations: ");
		System.out.println((double)iterations / trials);

		System.out.println("Average rate: ");
		System.out.println(rateSum / (double)trials);

	}

	public static void main(String[] args)
	{

		for (int i = 0; i < 10; i++) {
			RunSimulation(10000, 10);
		}

	}

}