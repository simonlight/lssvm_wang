package latent.lssvm;

import java.util.ArrayList;
import java.util.List;

import latent.LatentRepresentation;
import latent.LatentStructuralClassifier;
import solver.MosekSolver;
import struct.STrainingSample;
import util.VectorOp;
import fr.lip6.jkernelmachines.kernel.typed.DoubleLinear;
import fr.lip6.jkernelmachines.util.DebugPrinter;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;

public abstract class LSSVM<X,Y,H> implements LatentStructuralClassifier<X,Y,H>{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 6163630441026359472L;
	
	protected int optim = 1;
	protected double lambda = 1e-4;
	protected int epochsLatentMax = 5;
	protected int epochsLatentMin = 2;
	protected int cpmax = 50;
	protected int cpmin = 5;
	protected double epsilon = 1e-2;
	
	// debug
	DebugPrinter debug = new DebugPrinter();
	
	//svm hyperplane
	protected double[] w = null;
	protected int dim = 0;
	
	//linear kernel
	protected DoubleLinear linear = new DoubleLinear();
	
	protected abstract double[] psi(X x, Y y, H h);
	protected abstract double delta(Y y, Y yp);
	protected abstract void init(List<STrainingSample<LatentRepresentation<X,H>,Y>> l);
	/**
	 * loss augmented inference
	 * @param ts training sample
	 * @return (yp, hp)
	 * <p>
	 * res[0] : output (Y)
	 * </p>
	 * <p>
	 * res[1] : latent (H)
	 * </p>
	 */
	protected abstract Object[] lossAugmentedInference(STrainingSample<LatentRepresentation<X,H>,Y> ts);
	protected abstract H prediction(X x, Y y);

	@Override
	public void train(List<STrainingSample<LatentRepresentation<X,H>,Y>> l) {
		
		if(l.isEmpty())
			return;
	
		
		System.out.println("----------------------------------------------------------------------------------------");
		System.out.println("Train LSSVM - Mosek \tlambda: " + lambda + "\tepochsLatentMax " + epochsLatentMax + "\tepochsLatentMin " + epochsLatentMin); 
		System.out.println("epsilon= " + epsilon + "\t\tcpmax= " + cpmax + "\tcpmin= " + cpmin);
		init(l);
		if(optim == 1) {
			System.out.println("optim " + optim + " \t CCCP - Cutting-Plane 1 Slack");
		}
		System.out.println("----------------------------------------------------------------------------------------");
		
		w = new double[dim];
		
		long startTime = System.currentTimeMillis();
		if(optim == 1) {
			trainCCCPCuttingPlane1Slack(l);
		}
		else {
			System.out.println("ERROR Optim option invalid " + optim);
			System.exit(0);
		}
		long endTime = System.currentTimeMillis();
		System.out.println("Fin optim latent - Time learning= "+ (endTime-startTime)/1000 + "s");
		System.out.println("Evaluation after training " + evaluation(l));
		
		System.out.println("----------------------------------------------------------------------------------------");
	}
	
	/**
	 * Solve LSSVM optimization problem with CCCP. 
	 * Each iteration of the CCCP is solved with a cutting plane (1 slack formulation)
	 * @param l list of training samples
	 */
	protected void trainCCCPCuttingPlane1Slack(List<STrainingSample<LatentRepresentation<X,H>,Y>> l) {
		int el=0;
		double decrement = 0;
		double precObj = 0;
		while(el<epochsLatentMin || (el<=epochsLatentMax && decrement < 0)) {
			System.out.println("epoch latent " + el);
			// solve the convexified optimization problem 
			trainCCCPCP1Slack(l);
			double obj = primalObj(l);
			decrement = obj - precObj;
			System.out.println("obj= " + obj + "\tdecrement= " + decrement);
			precObj = obj;
			el++;
			
			// linearize the concave part
			for(STrainingSample<LatentRepresentation<X,H>,Y> ts : l){
				ts.input.h = prediction(ts.input.x,ts.output);
			}
		}
	}
	
	/**
	 * Solve the convexified optimization problem 
	 * @param l list of training samples
	 */
	protected void trainCCCPCP1Slack(List<STrainingSample<LatentRepresentation<X,H>,Y>> l) {
		
		double c = 1/lambda;
		int t=0;
		
		List<double[]> lg = new ArrayList<double[]>();
		List<Double> lc = new ArrayList<Double>();
		
		// Compute the initial cutting plane
		Object[] or = cuttingPlane(l);
		double[] gt = (double[]) or[0];
		double ct = (Double) or[1];
		
		lg.add(gt);
		lc.add(ct);
		
		double[][] gram = null;
		double xi=0;
		
		while(t<cpmin || (t<=cpmax && VectorOperations.dot(w,gt) < ct - xi - epsilon)) {
			
			System.out.print(".");
			if(t == cpmax) {
				System.out.print(" # max iter ");
			}
			
			// compute or update the gram matrix
			if(gram != null) {
				double[][] g = gram;
				gram = new double[lc.size()][lc.size()];
				for(int i=0; i<g.length; i++) {
					for(int j=0; j<g.length; j++) {
						gram[i][j] = g[i][j];
					}
				}
				for(int i=0; i<lc.size(); i++) {
					gram[lc.size()-1][i] = VectorOperations.dot(lg.get(lc.size()-1), lg.get(i));
					gram[i][lc.size()-1] = gram[lc.size()-1][i];
				}
				gram[lc.size()-1][lc.size()-1] += 1e-8;
			}
			else {
				gram = new double[lc.size()][lc.size()];
				for(int i=0; i<gram.length; i++) {
					for(int j=i; j<gram.length; j++) {
						gram[i][j] = VectorOperations.dot(lg.get(j), lg.get(i));
						gram[j][i] = gram[i][j];
						if(i==j) {
							gram[i][j] += 1e-8;
						}
					}
				}
			}
			// Solve the QP
			double[] alphas = MosekSolver.solveQP(gram, lc, c);
			//System.out.println("DualObj= " + (dot(alphas,lc.toArray(new Double[lc.size()])) - 0.5 * matrixProduct(alphas,gram)) + "\talphas " + Arrays.toString(alphas));
			xi = (VectorOp.dot(alphas,lc.toArray(new Double[lc.size()])) - matrixProduct(alphas,gram)) / c;
			
			// compute the new w
			w = new double[dim];
			for(int i=0; i<alphas.length; i++) {
				for(int d=0; d<dim; d++) {
					w[d] += alphas[i] * lg.get(i)[d];
				}
			}
			t++;

			// compute the new cutting plane model for the current w
			or = cuttingPlane(l);
			gt = (double[]) or[0];
			ct = (Double) or[1];
			
			lg.add(gt);
			lc.add(ct);
		}
		System.out.println(" Inner loop optimization finished.");
	}
	
	/**
	 * Compute the cutting plane for the current model w
	 * @param l list of training samples
	 * @return the cutting plane model
	 */
	public Object[] cuttingPlane(List<STrainingSample<LatentRepresentation<X,H>,Y>> l) {
		// compute g(t) and c(t)
		double[] gt = new double[w.length];
		double ct = 0;
		double n = l.size();
		
		for(int i=0; i<l.size(); i++){
			STrainingSample<LatentRepresentation<X,H>,Y> ts = l.get(i);			
			Object[] or = lossAugmentedInference(ts);
			Y yp = (Y)or[0];
			H hp = (H)or[1];
			// linear term of the cutting plane model
			ct += delta(ts.output, yp);
			// vector term of the cutting plane model
			double[] psi1 = psi(ts.input.x, yp, hp);
			double[] psi2 = psi(ts.input.x, ts.output, ts.input.h);
			for(int d=0; d<w.length; d++) {
				gt[d] += psi2[d] - psi1[d];
			}
		}
		ct /= n;
		
		for(int d=0; d<gt.length; d++) {
			gt[d] /= n;
		}
		
		Object[] res = new Object[2];
		res[0] = gt;
		res[1] = ct;
		return res;
	}
	
	/**
	 * compute the loss of the objective function <br />
	 * \sum_{i=1}^N max_{y,h} ( delta(yi,y) + <w, \psi(xi,y,h)> ) - max_hp <w,psi(xi,yi,hp)>
	 */
	protected double loss(List<STrainingSample<LatentRepresentation<X,H>,Y>> l) {
		double loss = 0;
		for(STrainingSample<LatentRepresentation<X,H>,Y> ts : l) {
			Object[] or = lossAugmentedInference(ts);
			Y yp = (Y)or[0];
			loss += delta(ts.output,yp);
		}
		loss /= l.size();
		return loss;	
	}
	
	/**
	 * Compute the primal objective value for the current model w
	 * @param l list of training samples
	 * @return primal objective value
	 */
	protected double primalObj(List<STrainingSample<LatentRepresentation<X,H>,Y>> l) {
		double obj = lambda * VectorOperations.dot(w,w)/2;
		double loss = loss(l);
		System.out.println("lambda*||w||^2= " + obj + "\t\tloss= " + loss);
		obj += loss;
		return obj;
	}
	
	public double evaluation(List<STrainingSample<LatentRepresentation<X,H>,Y>> l) {
		double delta = 0;
		for(STrainingSample<LatentRepresentation<X,H>,Y> ts : l) {
			Y yp = prediction(ts.input);
			delta += delta(ts.output,yp);
		}
		return delta;
	}
	
	protected double matrixProduct(double[] alphas, double[][] gram) {
		// alpha^T*Gramm*alpha
		double[] tmp = new double[alphas.length];
		// tmp = gram * alpha
		for(int i=0; i<gram.length; i++) {
			tmp[i] = VectorOperations.dot(gram[i],alphas);
		}
		double s = VectorOperations.dot(alphas,tmp);
		return s;
	}
	
	protected void showParameters() {
		System.out.println("----------------------------------------------------------------------------------------");
		
		System.out.println("----------------------------------------------------------------------------------------");
	}
	
	public String toString() {
		return "lssvm_optim_" + optim + "_lambda_" + lambda + "_epsilon_" + epsilon + "_cpmax_" + cpmax + "_cpmin_" + cpmin ;
	}
	
	public double getLambda() {
		return lambda;
	}
	
	/**
	 * Sets the hyperparameter lambda
	 * @param lambda
	 */
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}
	/**
	 * 
	 * @return the maximum number of CCCP iterations
	 */
	public int getEpochsLatentMax() {
		return epochsLatentMax;
	}
	/**
	 * Sets the maximum number of CCCP iterations
	 * @param epochsLatentMax  maximum number of CCCP iterations
	 */
	public void setEpochsLatentMax(int epochsLatentMax) {
		this.epochsLatentMax = epochsLatentMax;
	}
	/**
	 * 
	 * @return the minimum number of CCCP iterations
	 */
	public int getEpochsLatentMin() {
		return epochsLatentMin;
	}
	/**
	 * Sets the minimum number of CCCP iterations
	 * @param epochsLatentMin  minimum number of CCCP iterations
	 */
	public void setEpochsLatentMin(int epochsLatentMin) {
		this.epochsLatentMin = epochsLatentMin;
	}
	public double[] getW() {
		return w;
	}
	public void setW(double[] w) {
		this.w = w;
	}
	public int getOptim() {
		return optim;
	}
	/**
	 * Choose the method to solve the optimization problem <br />
	 * 1 -> dual with Mosek
	 * @param optim
	 */
	public void setOptim(int optim) {
		this.optim = optim;
	}
	/**
	 * 
	 * @return maximum number of cutting plane model
	 */
	public int getCpmax() {
		return cpmax;
	}
	/**
	 * Sets the maximum number of cutting plane model
	 * @param cpmax maximum number of cutting plane model
	 */
	public void setCpmax(int cpmax) {
		this.cpmax = cpmax;
	}
	/**
	 * 
	 * @return the minimum number of cutting plane model
	 */
	public int getCpmin() {
		return cpmin;
	}
	/**
	 * Sets the minimum number of cutting plane model
	 * @param cpmin minimum number of cutting plane model
	 */
	public void setCpmin(int cpmin) {
		this.cpmin = cpmin;
	}
	public double getEpsilon() {
		return epsilon;
	}
	public void setEpsilon(double epsilon) {
		this.epsilon = epsilon;
	}
}
