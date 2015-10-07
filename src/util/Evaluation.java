package util;

public class Evaluation<T>  implements Comparable<Evaluation<T>>
{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 791024170617779718L;
	
	public T sample;
	public double value;
	
	public Evaluation(T s, double v)
	{
		sample = s;
		value = v;
	}
	
	@Override
	public int compareTo(Evaluation<T> o) {
		if(o == null)
			return 0;
		return (int) Math.signum(o.value - value);
	}
}
