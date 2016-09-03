#ifndef BASECLASS_H
#define BASECLASS_H
class A
{
protected:
    double m_input;     // or use a pointer to a larger input object
public:
    virtual double f(double x) const = 0;
    void init(double input) { m_input=input; }
    virtual ~A() {};
};
#endif /* BASECLASS_H */
