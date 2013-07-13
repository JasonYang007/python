/** @package myCPP
*   C++ test program for Doxygen evaluation
*	\f[
*		\exp(i\pi)+1=0 
*	\f]
*	\f[
*		\int_{-\infty}^{\infty}\exp(-x^2)\,dx=\sqrt{2\pi}
*	\f]
*/
namespace MYCPP{

class Base{
public:
/** Base::hello function
*
*/
	hello() {print "Hello\n";}
};

class Derived: Base{
public:
/** Derived::Hello function
*
*/
	Hello() {Base::hello(); print "Derived hello\n";}
};

/** func1 function
* 
*/
int func1(void)
{
	return 0;
}

/** func2 function
*
*/
int func2(void)
{
	return func1();
}
/** func3 function
*
*/
int func3(void)
{
	func1();
	return func2();
}


} // end of name space
