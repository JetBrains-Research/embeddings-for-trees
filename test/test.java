public class Main {

    @Override
    protected Result execute(CLICommand cmd) throws Exception {
        return cmd.getExecutor(namenode).executeCommand(cmd.getCmd());
    }

}
