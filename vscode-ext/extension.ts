import * as vscode from "vscode";
import { exec } from "child_process";

export function activate(context: vscode.ExtensionContext) {

    const button = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Left,
        100
    );

    button.text = "$(play) Run tllm";
    button.tooltip = "Build and run tllm";
    button.command = "tllm.run";
    button.backgroundColor =    new vscode.ThemeColor("statusBarItem.errorBackground"
    );

    button.show();

    context.subscriptions.push(button);

    const runCommand = vscode.commands.registerCommand(
        "tllm.run",
        () => {

            const workspace =
                vscode.workspace.workspaceFolders?.[0];

            if (!workspace) {
                vscode.window.showErrorMessage(
                    "No workspace is open."
                );
                return;
            }

            const root = workspace.uri.fsPath;

            const terminal =
                vscode.window.createTerminal("tllm");

            terminal.show();

            terminal.sendText(
                `cd "${root}" && cmake --build build && ./build/app`
            );
        }
    );

    context.subscriptions.push(runCommand);
}

export function deactivate() {}